#!/usr/bin/env python3
"""
eqdsk_to_ingrid.py — build a ready-to-grid INGRID .yml from a freegs G-EQDSK.

Given a freegs-written g-file, this:
  1. reads the magnetic axis and divertor X-point(s) straight from the
     equilibrium (parse_geqdsk + lower_xpoints),
  2. derives target plates that span the exact psiN range INGRID will trace
     (target_finder.compute_plates) -- the cure for the recurring
     "one of the targets does not intersect one of the field lines" error,
  3. resolves the wall/limiter, and
  4. writes a new INGRID settings .yml by copying a TEMPLATE and overriding
     ONLY the per-equilibrium geometry (num_xpt, axis, X-point seeds, psi
     levels, target plates, limiter).  Physics knobs (psi levels, cell counts,
     tilts) come from the template, optionally overridden on the command line.

Why a template + overrides?  psi grid levels are NOT stored in a g-file:
freegs only fixes psiN=0 / psiN=1 via simag / sibry.  psi_1 (SOL depth),
psi_core, psi_pf_1/2 are gridding choices.  Whatever psi values end up in the
yaml are the SAME ones handed to the plate builder, so the plates always span
exactly what INGRID traces -- single source of truth.

INGRID only needs *close* seeds for rmagx/zmagx/rxpt/zxpt (it auto-refines via
root finders), and the topology is left for INGRID to auto-classify; we only
set num_xpt (1 for LSN/USN, 2 for snowflakes) and the X-point seeds.

Usage
-----
    python3 eqdsk_to_ingrid.py file.geqdsk --template base.yml [options]

    # also run INGRID through grid construction afterwards:
    python3 eqdsk_to_ingrid.py file.geqdsk --template base.yml --run
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.path import Path as MplPath
import yaml

from freegs_sf_creator import (parse_geqdsk, lower_xpoints, DEFAULT_WALL,
                               psi_sidecar_path)
from target_finder import PsiMap, compute_plates, plot_overlay, divertor_legs, trace_leg

# MAST-U divertor search windows (R0, R1, Z0, Z1) for the saddle finder.
LOWER_WINDOW = (0.4, 1.0, -1.6, -1.05)
UPPER_WINDOW = (0.4, 1.0, 1.05, 1.6)

# Default strike surface: the realistic MAST-U limiter polygon shared by
# freegs_sf_creator / target_finder (DEFAULT_WALL).  Consistency is essential:
# the SAME wall must feed (a) the psi sidecar's limiter-reachable spans, (b) the
# on-limiter plate carving, and (c) the grid, or the plates won't cover what
# INGRID traces.  (The legacy 5-point rectangular box under Reactors/MAST-U/
# SF-minus_exact_share dips to psiN~1.004 along its flat bottom between the
# snowflake nulls, starving the W2/E2 plates and stalling the tracer.)
DEFAULT_LIMITER = str(DEFAULT_WALL)

# INGRID package root used for --run/--tool imports.  Must be the build that
# supports the keys this pipeline emits (remove_upper_divertor) -- INGRID_Final,
# NOT the default-importable Ingrid_fixed (which rejects remove_upper_divertor).
DEFAULT_INGRID_ROOT = "/Users/sruiz/Dev/PhD/INGRID_Final/INGRID"


def nulls_in_band(d, pm, window, lo, hi, ntop=2):
    """X-points inside `window` whose psiN is in [lo, hi], as ((R, Z), psiN)
    sorted by |psiN - 1| (most separatrix-like first)."""
    found = [(p, pm.psin(*p)) for p in lower_xpoints(d, window=window, ntop=ntop)]
    found = [(p, pn) for p, pn in found if lo <= pn <= hi]
    found.sort(key=lambda t: abs(t[1] - 1.0))
    return found


def resolve_limiter(args, d, outdir):
    """Return (limiter_path, wall_xy).  Prefer the explicit --limiter file (the
    SAME polygon INGRID will grid against); else fall back to the rlim/zlim the
    g-file carries (freegs writes these when a wall is attached)."""
    if args.limiter != "eqdsk":
        path = Path(args.limiter)
        wall = np.loadtxt(path, delimiter=",")
        return path, wall
    lim = np.asarray(d["lim"])
    if lim.size == 0:
        raise SystemExit("--limiter eqdsk requested but the g-file carries no "
                         "rlim/zlim wall; pass --limiter <wall.txt> instead.")
    path = outdir / "limiter_from_eqdsk.txt"
    with open(path, "w") as f:
        for R, Z in lim:
            f.write(f"{R:.6f}, {Z:.6f}\n")
    print(f"  wrote limiter from g-file rlim/zlim -> {path}")
    return path, lim


def classify(d, pm, psi_1, tol, buffer, primary_pref, num_xpt_override):
    """Decide num_xpt, the X-point seeds and the plate-building jobs.

    A saddle counts as an *active* divertor X-point only if its psiN lies in
    ``[1 - tol, psi_1 + buffer]`` -- i.e. on the separatrix or within the SOL
    region actually being gridded.  A second X-point that sits *outside* psi_1
    is irrelevant to the grid, so the same equilibrium is single-null at a
    shallow psi_1 and double-null at a deep one.

    Returns (num_xpt, region, primary, secondary_or_None, labels)."""
    lo, hi = 1.0 - tol, psi_1 + buffer
    lower = [(p, pn, "lower") for p, pn in nulls_in_band(d, pm, LOWER_WINDOW, lo, hi)]
    upper = [(p, pn, "upper") for p, pn in nulls_in_band(d, pm, UPPER_WINDOW, lo, hi)]
    alln = lower + upper

    def fmt(ts):
        return [(round(p[0], 3), round(p[1], 3), round(pn, 4)) for p, pn, _ in ts]
    print(f"  X-points in psiN band [{lo:.4f}, {hi:.4f}]: "
          f"lower={fmt(lower)} upper={fmt(upper)}")

    if not alln:
        raise SystemExit(
            f"No divertor X-point found with psiN in [{lo:.4f}, {hi:.4f}]. "
            "Widen --xpt-tol, raise --psi-1, or check the equilibrium.")

    alln.sort(key=lambda t: abs(t[1] - 1.0))          # most separatrix-like first
    primary, _, region = alln[0]
    in_region = [t for t in alln if t[2] == region]

    n = len(in_region) if num_xpt_override == "auto" else int(num_xpt_override)
    n = max(1, min(n, 2))
    if n == 2 and len(in_region) < 2:
        print("  !! --num-xpt 2 requested but only one in-region null found; "
              "using num_xpt=1")
        n = 1

    if n == 2:
        a, b = in_region[0][0], in_region[1][0]       # a = psiN closest to 1
        if primary_pref == "left":
            primary, secondary = (a, b) if a[0] <= b[0] else (b, a)
        elif primary_pref == "right":
            primary, secondary = (a, b) if a[0] >= b[0] else (b, a)
        else:
            primary, secondary = a, b                 # auto: active null first
        return 2, region, primary, secondary, [(primary, "PRIM"), (secondary, "SEC")]

    return 1, region, primary, None, [(primary, "X")]


def _seg_intersect(a, b, c, d):
    """Intersection point of segments a-b and c-d, or None."""
    r = b - a
    s = d - c
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-12:
        return None
    qp = c - a
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return a + t * r
    return None


def _first_tile_crossing(leg, tiles):
    """Walk the traced leg polyline outward and return (name, strike_point,
    arclength) of the first tile it crosses, or (None, None, inf)."""
    best = (None, None, float("inf"))
    acc = 0.0
    for i in range(len(leg) - 1):
        a, b = np.asarray(leg[i], float), np.asarray(leg[i + 1], float)
        for name, T in tiles:
            for j in range(len(T) - 1):
                hit = _seg_intersect(a, b, np.asarray(T[j], float),
                                     np.asarray(T[j + 1], float))
                if hit is not None:
                    dist = acc + float(np.hypot(*(hit - a)))
                    if dist < best[2]:
                        best = (name, hit, dist)
        acc += float(np.hypot(*(b - a)))
    return best


def load_tiles(pdir):
    """Load every *_target.txt in `pdir` as a geometry pool: name -> (path, R/Z array)."""
    tiles = {}
    for f in sorted(Path(pdir).glob("*_target.txt")):
        arr = np.loadtxt(f, delimiter=",")
        if arr.ndim == 2 and len(arr) >= 2:
            tiles[f.stem.replace("_target", "")] = (f, arr)
    return tiles


def local_leg_dirs(pm, xpt, axis, r=0.04, n=360):
    """Separatrix-branch directions leaving `xpt`, found by the sign changes of
    psi-psi_x on a small circle of radius `r` (robust at the near-degenerate
    snowflake saddles where the Hessian asymptotes are unreliable).  Drops the
    branches heading back toward the magnetic axis."""
    psi_x = pm.psi(*xpt)
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    v = np.array([pm.psi(xpt[0] + r * np.cos(t), xpt[1] + r * np.sin(t)) - psi_x
                  for t in th])
    a_hat = np.array(axis) - np.array(xpt)
    a_hat = a_hat / (np.linalg.norm(a_hat) + 1e-30)
    dirs = []
    for i in range(n):
        j = (i + 1) % n
        if v[i] == 0 or v[i] * v[j] < 0:                  # sign change -> branch
            frac = v[i] / (v[i] - v[j]) if v[i] != v[j] else 0.5
            t = th[i] + frac * (2 * np.pi / n)
            dvec = np.array([np.cos(t), np.sin(t)])
            if dvec @ a_hat <= 0.3:                       # not core-ward
                dirs.append(dvec)
    return dirs


def assign_tiles_to_legs(pm, axis, primary, secondary, tiles, max_len=0.45):
    """Map plate slots W1/E1[/W2/E2] to the fixed tile each separatrix leg
    actually strikes.  Leg directions come from a small-circle psi scan, then
    each leg is marched a SHORT bounded arc (so it can't wander around the
    outboard snowflake lobe) and intersected with the tile pool; the lower-R
    strike is West.  Correct per equilibrium, regardless of orientation."""
    pool = [(nm, arr) for nm, (p, arr) in tiles.items()]
    nulls = [(primary, "1")]
    if secondary is not None:
        nulls.append((secondary, "2"))

    slot_to_name = {}
    strikes = {}
    for null, tag in nulls:
        cand = []
        for dvec in local_leg_dirs(pm, null, axis):
            leg, _ = trace_leg(pm, null, dvec, None, max_len=max_len)
            name, pt, dist = _first_tile_crossing(leg, pool)
            if name is not None:
                cand.append((dist, pt[0], name, pt))
        # keep the two distinct nearest tile strikes (by arclength)
        cand.sort(key=lambda c: c[0])
        chosen, used = [], set()
        for dist, R, name, pt in cand:
            if name in used:
                continue
            chosen.append((R, name, pt))
            used.add(name)
            if len(chosen) == 2:
                break
        chosen.sort(key=lambda c: c[0])                   # West = lower-R strike
        for we, c in zip(("W", "E"), chosen):
            slot_to_name[f"{we}{tag}"] = c[1]
            strikes[f"{we}{tag}"] = c[2]
        if len(chosen) < 2:
            print(f"  !! null {tag}: only {len(chosen)} distinct tile strike(s) found")
    return slot_to_name, strikes


def build_settings(template, d, num_xpt, primary, secondary, psi,
                   eqdsk_path, limiter_path, plate_files, strike, region):
    """Copy the template settings and override only the per-equilibrium
    geometry.  File paths are made ABSOLUTE with dir_settings left at '.', the
    same proven pattern as my_examples/MAST-U/mastu_sf75_grid.py."""
    s = template
    gs = s.setdefault("grid_settings", {})
    gs["num_xpt"] = num_xpt
    gs.setdefault("patch_generation", {})["strike_pt_loc"] = strike
    # For a lower-divertor grid, skip the upper-region patches: their poloidal
    # trace over the top can stall in a near-stagnation SOL, leaving an unset
    # E_spl and crashing ConstructGrid (see CLAUDE_INGRID_FINAL.md).
    if region == "lower":
        gs["remove_upper_divertor"] = True
    gs["rmagx"], gs["zmagx"] = float(d["rmaxis"]), float(d["zmaxis"])
    gs["rxpt"], gs["zxpt"] = float(primary[0]), float(primary[1])
    if num_xpt == 2:
        gs["rxpt2"], gs["zxpt2"] = float(secondary[0]), float(secondary[1])
    gs["psi_1"] = psi["psi_1"]
    gs["psi_core"] = psi["psi_core"]
    gs["psi_pf_1"] = psi["psi_pf_1"]
    if num_xpt == 2 and psi.get("psi_pf_2") is not None:
        gs["psi_pf_2"] = psi["psi_pf_2"]

    s["eqdsk"] = str(Path(eqdsk_path).resolve())
    s.setdefault("dir_settings", {})
    for k in ("eqdsk", "limiter", "target_plates", "patch_data"):
        s["dir_settings"][k] = "."

    # target plates: write all four keys (INGRID prepends paths for each);
    # unused ones (single null) are blanked so INGRID ignores them.
    tp = s.setdefault("target_plates", {})
    for name in ("W1", "E1", "W2", "E2"):
        entry = tp.setdefault(f"plate_{name}", {})
        if name in plate_files:
            entry["file"] = str(plate_files[name].resolve())
        else:
            entry["file"] = ""

    s["limiter"] = {"file": str(Path(limiter_path).resolve()),
                    "use_efit_bounds": False}
    return s


def _ingrid_worker(yaml_path, ingrid_root, stage):
    """Run INGRID in a child process up to `stage` ('patches' or 'grid') so a
    stalled line trace can be killed from the parent."""
    import sys
    if ingrid_root and ingrid_root not in sys.path:
        sys.path.insert(0, ingrid_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    from INGRID.ingrid import Ingrid

    ig = Ingrid(InputFile=str(yaml_path))
    ig.StartSetup()
    ig.AnalyzeTopology()
    ig.CreatePatches()
    if stage == "grid":
        ig.ConstructGrid(NewFig=False)
        print(f"==> INGRID identified '{ig.config}' and built the grid OK.")
    else:
        print(f"==> INGRID identified '{ig.config}' and built the patch map OK.")


def run_stage(yaml_path, ingrid_root, stage, timeout):
    """Run a worker stage in a watchdog'd subprocess.  Returns 'ok', 'fail' or
    'timeout'.  A SIGKILL (not SIGTERM) avoids a noisy 'Fatal Python error: GIL'
    when a stalled tracer is wedged in a C call."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")          # child re-imports cleanly (macOS default)
    p = ctx.Process(target=_ingrid_worker,
                    args=(str(yaml_path), ingrid_root, stage))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return "timeout"
    return "ok" if p.exitcode == 0 else "fail"


def run_ingrid(yaml_path, ingrid_root, timeout):
    """Single full grid run (for --run)."""
    print(f"\n==> Running INGRID on the generated settings (timeout {timeout}s) ...")
    status = run_stage(yaml_path, ingrid_root, "grid", timeout)
    if status == "timeout":
        raise SystemExit(
            f"# INGRID stalled (>{timeout}s) and was killed; the tracer wanders "
            "in a near-stagnation region.  Try --tune to search for parameters "
            "that grid, or a lower --psi-1.")
    if status == "fail":
        raise SystemExit("# INGRID exited with an error (traceback above). "
                         "Try --tune to search for working parameters.")


# --------------------------------------------------------------------------
# auto-tuner: search patch_generation / psi parameters until INGRID grids
# --------------------------------------------------------------------------
_PSI_KEYS = {"psi_1", "psi_2", "psi_core", "psi_pf_1", "psi_pf_2"}


def apply_overrides(settings, ov):
    """Apply a flat override dict onto grid_settings / patch_generation."""
    gs = settings["grid_settings"]
    pg = gs.setdefault("patch_generation", {})
    for k, v in ov.items():
        (gs if k in _PSI_KEYS else pg)[k] = v


def candidate_overrides(psi_1):
    """Parameter sets to try, cheap/likely first.

    Two families of knobs matter:
      * use_xpt*_W/E + xpt*_*_tilt -- these set how each X-point's SOL leg is
        drawn (e.g. B3_W from xpt2['W']), which fixes the START of the poloidal
        plate traces (A3_N -> WestPlate2 etc.).  When a plate trace wraps the
        whole SOL to the domain edge, this is the lever.
      * magx_tilt -- the core/SOL split angle; pair with a psi_1 back-off in
        case the SOL boundary is unreachable on this (large-domain) equilibrium.
    """
    tilt_vals = [-0.785, -0.4, 0.0, 0.4, 0.785]
    cands = []
    # toggle/steer the secondary-West and primary-East legs (the usual culprits
    # for the A3_N->WestPlate2 / F3_N->EastPlate1 plate traces)
    for uw in (True, False):
        for ue in (True, False):
            base = {"use_xpt2_W": uw, "use_xpt1_E": ue}
            cands.append(dict(base))
            if uw or ue:
                for t in tilt_vals:
                    ov = dict(base)
                    if uw:
                        ov["xpt2_W_tilt"] = t
                    if ue:
                        ov["xpt1_E_tilt"] = t
                    cands.append(ov)
    # then the core/SOL split angle, alone and with a psi_1 back-off
    for t in (0.0, -0.5, -0.7, 0.5):
        cands.append({"magx_tilt_1": t, "magx_tilt_2": t})
    for t in (-0.5, -0.7):
        for p1 in (round(psi_1 - 0.02, 4), round(psi_1 - 0.04, 4)):
            cands.append({"magx_tilt_1": t, "magx_tilt_2": t, "psi_1": p1})
    return cands


def fast_settings(settings):
    """A copy tuned for speed during the search: distortion off, coarse cells.
    Patch construction (where most failures fire) is resolution-independent;
    this only speeds the grid-confirmation step."""
    import copy
    s = copy.deepcopy(settings)
    gg = s["grid_settings"].setdefault("grid_generation", {})
    gg.setdefault("distortion_correction", {}).setdefault("all", {})["active"] = False
    gg["np_default"] = 3
    gg["nr_default"] = 3
    return s


def autotune(settings, out_path, ingrid_root, psi_1, patch_timeout, grid_timeout):
    """Search for patch_generation/psi parameters that let INGRID grid.  Each
    candidate is tested cheaply at CreatePatches; the first to pass is confirmed
    with a fast full grid.  On success the winning overrides are written onto
    the full-quality `settings` at `out_path`."""
    import copy
    cands = candidate_overrides(psi_1)
    base_test = fast_settings(settings)
    work = out_path.with_suffix(".tuning.yml")
    print(f"\n==> Auto-tuning: {len(cands)} candidate parameter sets "
          f"(patch timeout {patch_timeout}s, grid timeout {grid_timeout}s)")

    winner = None
    for i, ov in enumerate(cands, 1):
        test = copy.deepcopy(base_test)            # pristine each time (no leak)
        apply_overrides(test, ov)
        with open(work, "w") as f:
            yaml.safe_dump(test, f, default_flow_style=False, sort_keys=False)
        tag = ", ".join(f"{k}={v}" for k, v in ov.items())
        st = run_stage(work, ingrid_root, "patches", patch_timeout)
        print(f"  [{i:2d}/{len(cands)}] {tag:<45} patches -> {st}")
        if st != "ok":
            continue
        st2 = run_stage(work, ingrid_root, "grid", grid_timeout)
        print(f"             {'':<45} full grid -> {st2}")
        if st2 == "ok":
            winner = ov
            break

    try:
        work.unlink()
    except OSError:
        pass

    if winner is None:
        raise SystemExit("# Auto-tune exhausted all candidates without a full "
                         "grid. Inspect the failures above; widen the sweep "
                         "(magx tilts / psi) if needed.")

    apply_overrides(settings, winner)
    with open(out_path, "w") as f:
        yaml.safe_dump(settings, f, default_flow_style=False, sort_keys=False)
    print(f"\n==> SUCCESS: INGRID grids with {winner}")
    print(f"  final full-quality settings -> {out_path}")
    return winner


def main():
    ap = argparse.ArgumentParser(
        description="Build an INGRID .yml (axis, X-points, psi seeds, target "
                    "plates, limiter) from a freegs G-EQDSK.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("eqdsk", help="input freegs G-EQDSK file")
    ap.add_argument("--template", required=True,
                    help="base INGRID .yml supplying physics knobs (psi levels, "
                         "cell counts, tilts, integrator settings)")
    ap.add_argument("--limiter", default=DEFAULT_LIMITER,
                    help="limiter polygon (R,Z csv) used as the strike surface "
                         "(default) and as the grid wall. Pass 'eqdsk' to use "
                         "the g-file's rlim/zlim.")
    ap.add_argument("--psi-1", type=float, default=None,
                    help="override SOL boundary psiN (yaml psi_1)")
    ap.add_argument("--psi-core", type=float, default=None,
                    help="override core boundary psiN (yaml psi_core)")
    ap.add_argument("--psi-pf1", type=float, default=None,
                    help="override primary private-flux psiN (yaml psi_pf_1)")
    ap.add_argument("--psi-pf2", type=float, default=None,
                    help="override secondary private-flux psiN (yaml psi_pf_2)")
    ap.add_argument("--no-sidecar", action="store_true",
                    help="ignore the <eqdsk>.psi.json sidecar (limiter-derived "
                         "psi levels from freegs_sf_creator); use template/CLI only")
    ap.add_argument("--plates-dir", default=None,
                    help="directory holding the (fixed, machine-geometry) "
                         "W1/E1[/W2/E2]_target.txt tile files "
                         "(default: the template's directory)")
    ap.add_argument("--limiter-strike", action="store_true",
                    help="strike field lines on the limiter wall instead of "
                         "target plates (strike_pt_loc=limiter); note this does "
                         "not bound the SF lower divertor on MAST-U")
    ap.add_argument("--literal-plates", action="store_true",
                    help="trust tile FILENAMES as W1/E1/W2/E2 instead of "
                         "assigning each tile to the leg it strikes (default)")
    ap.add_argument("--auto-plates", action="store_true",
                    help="derive plates from the equilibrium via target_finder "
                         "instead of using the tile files (heuristic; verify the "
                         "overlay PNG)")
    ap.add_argument("--margin", type=float, default=0.012,
                    help="[--auto-plates] extra psiN each side of the plates so "
                         "traces hit with room to spare")
    ap.add_argument("--plate-points", type=int, default=2,
                    help="[--auto-plates] points per plate; 2 = the developer's "
                         "straight-line route (psi_pf/psi_sol endpoints only)")
    ap.add_argument("--float-plates", action="store_true",
                    help="[--auto-plates] float perpendicular plates a back-off "
                         "inside the wall instead of carving them onto the "
                         "limiter (default is on-limiter for SN and snowflake)")
    ap.add_argument("--xpt-tol", type=float, default=0.05,
                    help="lower psiN tolerance (1-tol) for counting a saddle as "
                         "an active divertor X-point; the upper bound is psi_1")
    ap.add_argument("--xpt-buffer", type=float, default=0.005,
                    help="psiN slack above psi_1 when admitting an X-point")
    ap.add_argument("--num-xpt", choices=("auto", "1", "2"), default="auto",
                    help="force the number of divertor X-points instead of "
                         "auto-detecting from the psiN band")
    ap.add_argument("--primary", choices=("auto", "left", "right"), default="auto",
                    help="which lower null is the active/primary one (snowflakes)")
    ap.add_argument("--outdir", default=None,
                    help="dir for plates/overlay/yaml (default <eqdsk>_ingrid)")
    ap.add_argument("--out", default=None,
                    help="output .yml path (default <outdir>/<eqdsk stem>.yml)")
    ap.add_argument("--run", action="store_true",
                    help="run INGRID through ConstructGrid after writing the yaml")
    ap.add_argument("--tune", action="store_true",
                    help="search patch_generation/psi parameters until INGRID "
                         "grids (tests cheaply at CreatePatches, confirms the "
                         "winner with a full grid)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds before a stalled INGRID grid run is killed")
    ap.add_argument("--patch-timeout", type=float, default=120.0,
                    help="[--tune] seconds before a candidate's CreatePatches "
                         "test is killed")
    ap.add_argument("--ingrid-root", default=DEFAULT_INGRID_ROOT,
                    help="path to the INGRID package root (for --run/--tune "
                         "imports). Defaults to INGRID_Final, which supports the "
                         "remove_upper_divertor key this pipeline emits; the "
                         "default-importable Ingrid_fixed does not.")
    args = ap.parse_args()

    eqdsk_path = Path(args.eqdsk).resolve()
    d = parse_geqdsk(str(eqdsk_path))
    pm = PsiMap(d)
    axis = (float(d["rmaxis"]), float(d["zmaxis"]))

    outdir = Path(args.outdir) if args.outdir else \
        eqdsk_path.parent / f"{eqdsk_path.stem}_ingrid"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.template) as f:
        template = yaml.safe_load(f)
    tgs = template.get("grid_settings", {})

    # psi sidecar (foo.psi.json next to the g-file): the limiter-derived levels
    # freegs_sf_creator wrote.  Loaded if present unless --no-sidecar.
    side = {}
    sidecar = psi_sidecar_path(eqdsk_path)
    if not args.no_sidecar and sidecar.exists():
        with open(sidecar) as f:
            side = json.load(f)
        print(f"psi sidecar: {sidecar} -> {side}")

    # psi levels, precedence CLI override > sidecar > template (single source of
    # truth: whatever lands here is also handed to the plate builder)
    def pick(cli, key):
        if cli is not None:
            return cli
        if key in side:
            return side[key]
        return tgs.get(key)

    psi = {
        "psi_1":    pick(args.psi_1,    "psi_1"),
        "psi_core": pick(args.psi_core, "psi_core"),
        "psi_pf_1": pick(args.psi_pf1,  "psi_pf_1"),
        "psi_pf_2": pick(args.psi_pf2,  "psi_pf_2"),
    }
    if psi["psi_1"] is None or psi["psi_pf_1"] is None:
        raise SystemExit("psi_1 and psi_pf_1 must come from the template or "
                         "--psi-1/--psi-pf1; neither was found.")

    print(f"eqdsk     : {eqdsk_path}")
    print(f"template  : {args.template}")
    print(f"mag axis  : R={axis[0]:.4f} Z={axis[1]:+.4f}")
    print(f"psi levels: psi_1={psi['psi_1']} psi_core={psi['psi_core']} "
          f"psi_pf_1={psi['psi_pf_1']} psi_pf_2={psi['psi_pf_2']}")

    limiter_path, wall = resolve_limiter(args, d, outdir)
    limiter = MplPath(wall)

    num_xpt, region, primary, secondary, labels = classify(
        d, pm, psi["psi_1"], args.xpt_tol, args.xpt_buffer,
        args.primary, args.num_xpt)
    print(f"  -> num_xpt={num_xpt} ({region} divertor)")
    print(f"  primary  X: R={primary[0]:.4f} Z={primary[1]:+.4f}")
    if secondary is not None:
        print(f"  secondary X: R={secondary[0]:.4f} Z={secondary[1]:+.4f}")

    # ---- strike target -------------------------------------------------
    # Default: strike on the fixed MAST-U tile files in the template's dir.
    # --auto-plates derives plates heuristically; --limiter-strike strikes on
    # the wall instead.
    if args.auto_plates:
        strike = "target_plates"
        # Size all plates down to the smallest configured psi_pf: INGRID traces
        # the DEEPEST private-flux level to every plate, so a plate that stops
        # short lets the field line slip to the boundary ("missed target").
        psi_pf_lo = min(v for v in (psi["psi_pf_1"], psi["psi_pf_2"])
                        if v is not None)
        if num_xpt == 2:
            jobs = ((primary, secondary, "1", psi_pf_lo),
                    (secondary, primary, "2", psi_pf_lo))
        else:
            jobs = ((primary, None, "1", psi_pf_lo),)
        # carve every plate out of the limiter so the grid ends on the physical
        # boundary (target_finder.build_plate_on_limiter) -- single-null AND
        # snowflake now share this convention; --float-plates restores the old
        # floating perpendicular plates for debugging
        plates = compute_plates(pm, axis, jobs, psi["psi_1"], args.margin,
                                limiter, outdir, npts=args.plate_points,
                                wall=wall, on_limiter=not args.float_plates)
        plot_overlay(d, wall, labels, plates, psi["psi_1"], outdir,
                     f"eqdsk_to_ingrid: {eqdsk_path.name}")
        plate_files = {name: outdir / f"{name}_target.txt" for name in plates}
    elif args.limiter_strike:
        strike = "limiter"
        plate_files = {}
        print("  strike target: MAST-U limiter (strike_pt_loc=limiter, "
              "no separate target plates)")
    else:
        strike = "target_plates"
        pdir = Path(args.plates_dir) if args.plates_dir \
            else Path(args.template).resolve().parent
        tiles = load_tiles(pdir)
        if not tiles:
            raise SystemExit(f"No *_target.txt tile files found in {pdir} "
                             "(pass --plates-dir, or --limiter-strike).")
        if args.literal_plates:
            # trust the filenames as the W1/E1/W2/E2 labels
            wanted = ("W1", "E1", "W2", "E2") if num_xpt == 2 else ("W1", "E1")
            plate_files = {n: tiles[n][0] for n in wanted if n in tiles}
            print(f"  using target plates (literal names) from {pdir}: "
                  f"{sorted(plate_files)}")
        else:
            # assign each fixed tile to the leg it actually strikes
            slot_to_name, strikes = assign_tiles_to_legs(
                pm, axis, primary, secondary, tiles)
            plate_files = {slot: tiles[name][0]
                           for slot, name in slot_to_name.items()}
            print(f"  tile->leg assignment (pool {sorted(tiles)} in {pdir}):")
            for slot in ("W1", "E1", "W2", "E2"):
                if slot in slot_to_name:
                    R, Z = strikes[slot]
                    print(f"    {slot} <- {slot_to_name[slot]}_target.txt "
                          f"(strike R={R:.3f} Z={Z:+.3f})")
        if "W1" not in plate_files or "E1" not in plate_files:
            raise SystemExit(
                f"Could not assign at least W1/E1 from tiles in {pdir}.")

    settings = build_settings(template, d, num_xpt, primary, secondary, psi,
                              eqdsk_path, limiter_path, plate_files, strike, region)

    out = Path(args.out) if args.out else outdir / f"{eqdsk_path.stem}.yml"
    with open(out, "w") as f:
        yaml.safe_dump(settings, f, default_flow_style=False, sort_keys=False)
    print(f"\n  wrote INGRID settings -> {out}")

    if args.tune:
        autotune(settings, out, args.ingrid_root, psi["psi_1"],
                 args.patch_timeout, args.timeout)
    elif args.run:
        run_ingrid(out, args.ingrid_root, args.timeout)


if __name__ == "__main__":
    main()
