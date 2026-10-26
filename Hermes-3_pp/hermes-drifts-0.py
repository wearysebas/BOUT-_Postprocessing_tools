#!/usr/bin/env python3
"""
hermes-drifts-0.py — first-look plots for Hermes-3 runs with drifts
(``relax_potential`` + electron ``evolve_momentum``).

Implements the plotting plan in ``Hermes-Final/CLAUDE_DRIFTS.md`` §10:

    Tier 0  Has phi converged?        phi at probe points vs t, max/rms dphi/dt,
                                      target particle/power flows vs t, totals
    Tier 1  2D maps                   phi, phi/Te, E_r, E_pol, v_E,r, v_E,pol,
                                      radial ExB particle flux,
                                      |Gamma_ExB| / |Gamma_anomalous|, j_par,
                                      div J_dia, Ne, Te  (separatrices and
                                      branch cuts overlaid)
            1D cuts of components     radial cuts (OMP + each leg entrance) and
                                      poloidal cuts (first SOL ring + last ring
                                      inside) of E_r, E_pol, v_E,r, v_E,pol,
                                      n v_E,r
    Tier 2  1D profiles               OMP radial, per-target, along one ring
    Tier 3  Per-strike-point budget   power, particle flow, current budget:
                                      j_par and J_dia into each target plus
                                      J_dia through the radial boundaries
                                      (sum should be ~0)
            All-case summary          power and particle flow per strike point
                                      for every case, both runs
    Tier 4  Drifts vs no drifts       (with --compare) side-by-side Ne/Te,
                                      Delta maps, Delta target profiles;
                                      Tiers 2/3 overlay both runs automatically

Figures go to ``<outdir>/Drifts/<case>/`` or ``<outdir>/No_drifts/<case>/``
depending on ``--drifts``; the all-case summary to ``<outdir>/<Drifts|No_drifts>/``.

--------------------------------------------------------------------------------
Reconstructed fields (not in the dumps)
--------------------------------------------------------------------------------
Anomalous radial particle flow, when ``pf<ion>_tot_xlow`` is absent: the x-face
flow of ``Div_a_Grad_perp_upwind_flows`` (src/div_ops.cxx), evaluated from the
dumped ``anomalous_D_<ion>`` and ``N<ion>``

    flow_{i+1/2} = -[(J g11)_i + (J g11)_{i+1}] (N_{i+1} - N_i) / (dx_i + dx_{i+1})
                   * D_upwind * dy dz

Diamagnetic current, as ``RelaxPotential::calculateDivJdia``: summed over the
charged species with a pressure,

    J_dia = sum_s P_s Curlb_B,   Curlb_B^{x,y} = 2 bxcv{x,y} / B   (SI)
    div J_dia = ( d/dx(J J_dia^x) + d/dy(J J_dia^y) ) / J   (BOUT Div(Vector3D))

--------------------------------------------------------------------------------
Reuse
--------------------------------------------------------------------------------
Nothing is re-implemented that already exists in the toolchain. Both modules are
imported by path (their filenames are not legal identifiers):

  * ``Hermes-3_postprocessing.py``            polygon_plot, _proc_index
  * ``hermes-heat-pp-method-comparison.py``   load_run (time-aligned xHermes open
                                              + y-guarded view), _guard_map,
                                              _tavg, method1_energy_flow,
                                              sheath_potential_check

So the two gotchas documented in the heat script apply and are handled the same
way here: upper-target faces are read from the y-guarded view, and 2D arrays
are handed to ``polygon_plot`` with the FULL radial extent (x guards included).

--------------------------------------------------------------------------------
ExB velocity: same formula as Hermes-3
--------------------------------------------------------------------------------
Hermes reads the metric from the grid file (``recalculate_metric = false`` is
the default) and advects with ``Div_n_bxGrad_f_B_XPPM`` (src/div_ops.cxx),
which in axisymmetry (d/dz = 0) uses the contravariant velocities

    v^x =  g11 g23 / B^2 * dphi/dy          (flux through x faces: n v^x J dy dz)
    v^y = -g11 g23 / B^2 * dphi/dx          (flux through y faces: n v^y J dx dz)

This script evaluates exactly those, using g11, g23, Bxy, J, dx, dy from the
grid file, so the SIGN CONVENTION IS THE CODE'S, not a hand-derived one. The
physical components plotted are

    v_E,r   = v^x / sqrt(g11)      [m/s]   along grad(psi)  (+ = increasing x)
    v_E,pol = v^y * h_y            [m/s]   poloidal, + = increasing y index
    E_r     = -sqrt(g11) dphi/dx   [V/m]
    E_pol   = -(1/h_y)   dphi/dy   [V/m]

with h_y = hthe (poloidal length per unit y). For the standard tokamak metric
(g23 = -Bt/(hthe Bp R)) these reduce to v_E,r = Bt E_pol / B^2 and
v_E,pol = -Bt E_r / B^2, i.e. E x B / B^2.

y-derivatives are one-sided at every branch cut (jyseps*, ny_inner), because
index neighbours across a cut are not spatial neighbours. The script also
prints a metric sanity check (hthe*dy and dx/sqrt(g11) vs the actual cell-centre
spacing from Rxy/Zxy); if those ratios are far from 1 the grid's metric and
geometry disagree and the physical magnitudes should not be trusted.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
    python hermes-drifts-0.py <sims_dir> <grid_dir> [--only SF45 ...] [options]

    Every SF<angle>/SN folder in sims_dir is paired with its grid in grid_dir
    (discover_cases from SF_heat_transport_pp_xhermes_withfluxes.py), exactly
    as in hermes-heat-pp-method-comparison.py.

    # every drift case, files only
    python hermes-drifts-0.py CVODE_test grids/Hermes-3/sf-new-limiter

    # drifts vs the matched no-drift runs (cases matched by folder name)
    python hermes-drifts-0.py CVODE_test grids/Hermes-3/sf-new-limiter \\
        --only SF45 --compare No_Drifts

    # the no-drift runs on their own, into Post_Processing/No_drifts
    python hermes-drifts-0.py No_Drifts grids/Hermes-3/sf-new-limiter --drifts False

    # only the convergence check, shown on screen
    python hermes-drifts-0.py CVODE_test grids/Hermes-3/sf-new-limiter \\
        --only SF45 --tiers 0 --show
"""

import argparse
import glob
import importlib.util
import os
import sys
import traceback
import types

import dask
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


BANNER = "hermes-drifts-0"

QE = 1.602176634e-19   # elementary charge [C]
MP = 1.672621898e-27   # proton mass [kg]
ME = 9.1093837015e-31  # electron mass [kg]

_HERE = os.path.dirname(os.path.abspath(__file__))

_PP_CANDIDATES = [
    os.path.join(_HERE, "Hermes-3_postprocessing.py"),
    os.path.expanduser("~/Dev/My_working_scripts/BOUT++_Post_Processing/"
                       "Hermes-3_postprocessing.py"),
]
_HEAT_CANDIDATES = [
    os.path.join(_HERE, "hermes-heat-pp-method-comparison.py"),
    os.path.expanduser("~/Dev/My_working_scripts/Hermes-3_pp/"
                       "hermes-heat-pp-method-comparison.py"),
]

# netCDF4/HDF5 is not thread-safe and xarray closes garbage-collected files
# without its lock; with dask's threaded scheduler that close races the next
# case's reads -> "NetCDF: HDF error" + segfault. Arrays are tiny: run serially.
dask.config.set(scheduler="synchronous")

RUN_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def _load_module(modname, candidates, explicit=None):
    """Import a toolchain script by file path (see module docstring)."""
    for path in ([explicit] if explicit else []) + candidates:
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"[{BANNER}] Loaded {modname} from {path}")
            return mod
    sys.exit(f"[{BANNER}] Could not find {modname} (looked in "
             f"{([explicit] if explicit else []) + candidates}).")


# ══════════════════════════════════════════════════════════════════════════════
# Loading: per-rank time alignment
# ══════════════════════════════════════════════════════════════════════════════

def _aligned_datasets(path, pp):
    """
    Per-rank dump files truncated to the last step common to EVERY rank.

    Same two-step truncation as ``open_aligned_boutdataset`` in
    Hermes-3-Conv-Test.py / Hermes-3_postprocessing.py, but returning the
    aligned list so it can go through xHermes (for unit conversion):

      1. clip every file to the shortest ``t`` length;
      2. clip again at the first index where any rank's ``t_array`` differs
         from rank 0's. A rank killed mid-write can have the right ``t`` length
         but a NaN last time; NaN != NaN, so this catches it too.
    """
    files = sorted(glob.glob(os.path.join(path, "BOUT.dmp.*.nc")), key=pp._proc_index)
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp files in {path!r}")

    datasets = [xr.open_dataset(f, engine="netcdf4") for f in files]

    tlens = [ds.sizes.get("t", 0) for ds in datasets]
    if min(tlens) == 0:
        raise ValueError(
            "At least one dump file has no time dimension / no output steps; "
            "cannot align."
        )

    min_t = min(tlens)
    if len(set(tlens)) > 1:
        max_t = max(tlens)
        n_short = sum(1 for n in tlens if n < max_t)
        print(
            f"[{BANNER}] Time-step mismatch across dump files: "
            f"lengths range {min_t}..{max_t} "
            f"({n_short}/{len(tlens)} ranks short of the maximum). "
            f"Truncating all files to the last common step (t index 0..{min_t - 1})."
        )
    else:
        print(
            f"[{BANNER}] All {len(files)} dump files agree on "
            f"{min_t} time steps; no truncation needed."
        )

    datasets = [ds.isel(t=slice(0, min_t)) for ds in datasets]

    t_ref = datasets[0]["t_array"].values
    good = np.ones(min_t, dtype=bool)
    for ds in datasets[1:]:
        good &= ds["t_array"].values == t_ref
    n_good = int(np.argmin(good)) if not good.all() else min_t
    if n_good < min_t:
        print(
            f"[{BANNER}] t_array mismatch across ranks from t index "
            f"{n_good}; truncating to 0..{n_good - 1}."
        )
        datasets = [ds.isel(t=slice(0, n_good)) for ds in datasets]

    return datasets


def _use_aligned_open(heat, pp):
    """
    Route the heat module's ``xhermes.open`` through ``_aligned_datasets`` so
    ``load_run`` (both its geometry view and its y-guarded view) survives
    per-rank time mismatches. The heat script itself is left untouched.
    """
    real = heat.xhermes

    def _open(path, **kwargs):
        return real.open_hermesdataset(
            datapath=_aligned_datasets(path, pp),
            inputfilepath=os.path.join(path, "BOUT.settings"), **kwargs)

    heat.xhermes = types.SimpleNamespace(open=_open, open_hermesdataset=real.open_hermesdataset)


# ══════════════════════════════════════════════════════════════════════════════
# Geometry
# ══════════════════════════════════════════════════════════════════════════════

def _cut_interfaces(grid, ny, ny_inner):
    """
    y-interfaces (between j and j+1) that are NOT spatial neighbours for at
    least some x: every jyseps* branch cut plus the ny_inner target gap.
    Treating them as cuts for all x only downgrades a centred difference to a
    one-sided one in the SOL, which is harmless.
    """
    cuts = set()
    for name in ("jyseps1_1", "jyseps2_1", "jyseps1_2", "jyseps2_2"):
        if name in grid:
            c = int(grid[name].values)
            if 0 <= c < ny - 1:
                cuts.add(c)
    if 0 < ny_inner < ny:
        cuts.add(ny_inner - 1)
    return sorted(cuts)


def _y_neighbours(R, Z, rm, zm, cuts):
    """
    (up, dn): y index of the cell across the yhigh / ylow face of (i, j) on the
    same ring, or -1 at a target.

    With the gridue corner mesh, the neighbour across a face is the cell that
    shares both of its corners, so branch cuts connect to their true spatial
    neighbours (as BOUT's communication does). Without it, index neighbours
    with every cut treated as a boundary.
    """
    nx, ny = R.shape
    up = np.tile(np.arange(ny) + 1, (nx, 1))
    dn = np.tile(np.arange(ny) - 1, (nx, 1))
    up[:, -1] = -1
    dn[:, 0] = -1
    for c in cuts:
        up[:, c] = -1
        dn[:, c + 1] = -1
    if rm is None or zm is None:
        return up, dn

    scale = float(np.nanmax(np.abs(R)))
    key = lambda r, z: (round(r / scale, 7), round(z / scale, 7))  # noqa: E731
    cen = np.stack([rm[..., 0].ravel(), zm[..., 0].ravel()], axis=1)
    cells = {}
    for i in range(nx):
        for j in range(ny):
            d = np.hypot(cen[:, 0] - R[i, j], cen[:, 1] - Z[i, j])
            k = int(np.argmin(d))
            if d[k] < 1e-6 * scale:
                cells[np.unravel_index(k, rm.shape[:2])] = (i, j)

    def face(g, a, b):
        return frozenset((key(rm[g][a], zm[g][a]), key(rm[g][b], zm[g][b])))

    by_face = {}
    for g in cells:
        for a, b in ((1, 3), (2, 4)):
            by_face.setdefault(face(g, a, b), []).append(g)

    glist = list(cells)
    corners = {(a, b): np.array([[[rm[h][a], zm[h][a]], [rm[h][b], zm[h][b]]] for h in glist])
               for a, b in ((1, 3), (2, 4))}

    def across(g, a, b):
        other = [h for h in by_face[face(g, a, b)]
                 if h != g and cells[h][0] == cells[g][0]]
        if other:
            return cells[other[0]][1]
        mine = np.array([[rm[g][a], zm[g][a]], [rm[g][b], zm[g][b]]])
        opp = corners[(2, 4) if a == 1 else (1, 3)]
        d = np.minimum(np.max(np.linalg.norm(opp - mine, axis=-1), axis=-1),
                       np.max(np.linalg.norm(opp - mine[::-1], axis=-1), axis=-1))
        k = int(np.argmin(d))
        if (glist[k] != g and cells[glist[k]][0] == cells[g][0]
                and d[k] < 1e-2 * np.linalg.norm(mine[0] - mine[1])):
            return cells[glist[k]][1]
        return -1

    hi = np.full((nx, ny), -1)
    lo = np.full((nx, ny), -1)
    for g, (i, j) in cells.items():
        hi[i, j] = across(g, 1, 3)
        lo[i, j] = across(g, 2, 4)
    matched = np.zeros((nx, ny), bool)
    for i, j in cells.values():
        matched[i, j] = True
    if np.mean(hi[matched] == up[matched]) < np.mean(lo[matched] == up[matched]):
        hi, lo = lo, hi
    rows = np.where(matched.any(axis=1))[0]
    for i in range(nx):
        src = i if matched[i].all() else rows[np.argmin(np.abs(rows - i))]
        up[i], dn[i] = hi[src], lo[src]
    return up, dn


def _ddy(f, dy, nbr):
    """dF/dy in index space, centred (as BOUT's DDY: /2dy) across true y
    neighbours ``nbr = (up, dn)``, one-sided at the targets."""
    up, dn = nbr
    rows = np.arange(f.shape[0])[:, None]
    fu = np.where(up >= 0, f[rows, np.clip(up, 0, None)], np.nan)
    fd = np.where(dn >= 0, f[rows, np.clip(dn, 0, None)], np.nan)
    both = (up >= 0) & (dn >= 0)
    out = np.where(both, (fu - fd) / (2.0 * dy),
                   np.where(up >= 0, (fu - f) / dy, (f - fd) / dy))
    out[(up < 0) & (dn < 0)] = np.nan
    return out


def _ddx(f, dx):
    """dF/dx in index space, centred, one-sided at the radial ends."""
    out = np.empty_like(f, dtype=float)
    out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx[1:-1])
    out[0] = (f[1] - f[0]) / dx[0]
    out[-1] = (f[-1] - f[-2]) / dx[-1]
    return out


def load_geometry(grid_file, run):
    """
    Read the grid fields Hermes actually uses (metric + B) and the corner mesh.

    Grid fields live on dims (x2, y2) with shape (nx, ny) — full radial extent
    including x guards, which is also what the xHermes interior-frame ``ds``
    carries, so they line up index for index.
    """
    g = xr.open_dataset(grid_file, engine="netcdf4")
    nx, ny, ny_inner = run["nx"], run["ny"], run["ny_inner"]

    def gv(name):
        if name not in g:
            return None
        a = np.asarray(g[name].values, dtype=float)
        return a if a.shape == (nx, ny) else None

    R, Z = gv("Rxy"), gv("Zxy")
    Bp, Bt, B = gv("Bpxy"), gv("Btxy"), gv("Bxy")
    hthe, dx, dy = gv("hthe"), gv("dx"), gv("dy")
    g11, g23, J = gv("g11"), gv("g23"), gv("J")
    bxcvx, bxcvy = gv("bxcvx"), gv("bxcvy")
    if R is None or Z is None or dx is None or dy is None:
        sys.exit(f"[{BANNER}] Grid lacks Rxy/Zxy/dx/dy with shape ({nx}, {ny}).")
    if B is None:
        B = np.sqrt(Bp ** 2 + Bt ** 2)
    cuts = _cut_interfaces(g, ny, ny_inner)
    rm = g["rm"].values if "rm" in g else None
    zm = g["zm"].values if "zm" in g else None
    nbr = _y_neighbours(R, Z, rm, zm, cuts)
    linked = [(c, int(np.sum(nbr[0][:, c] >= 0))) for c in cuts]
    print(f"[{BANNER}] y-faces linked across each cut (x points with a neighbour, "
          f"of {nx}): " + ", ".join(f"after j={c}: {n}" for c, n in linked))

    if hthe is None:
        # Poloidal length per unit y from the cell-centre positions.
        hthe = np.hypot(_ddy(R, dy, nbr), _ddy(Z, dy, nbr))
        print(f"[{BANNER}] No hthe in grid; using |d(R,Z)/dy| from Rxy/Zxy.")
    if g11 is None:
        g11 = (R * Bp) ** 2
        print(f"[{BANNER}] No g11 in grid; using (R Bp)^2.")
    if g23 is None:
        g23 = -Bt / (hthe * Bp * R)
        print(f"[{BANNER}] No g23 in grid; using -Bt/(hthe Bp R) "
              f"(standard tokamak metric).")
    if J is None:
        J = hthe / Bp

    # ── Metric vs geometry sanity check ──────────────────────────────────────
    sx = slice(run["MXG"], nx - run["MXG"])
    dl_y = np.hypot(np.diff(R, axis=1), np.diff(Z, axis=1))          # j|j+1
    hy_mid = 0.5 * (hthe[:, 1:] * dy[:, 1:] + hthe[:, :-1] * dy[:, :-1])
    ok = np.ones(ny - 1, bool)
    ok[cuts] = False
    ratio_y = (hy_mid / dl_y)[sx][:, ok]
    dl_x = np.hypot(np.diff(R, axis=0), np.diff(Z, axis=0))
    hx_mid = 0.5 * (dx[1:] / np.sqrt(g11[1:]) + dx[:-1] / np.sqrt(g11[:-1]))
    ratio_x = (hx_mid / dl_x)[run["MXG"]:nx - run["MXG"] - 1]
    ry, rx = np.nanmedian(ratio_y), np.nanmedian(ratio_x)
    print(f"[{BANNER}] Metric check (median metric length / Rxy-Zxy spacing): "
          f"poloidal {ry:.3f}, radial {rx:.3f}   (both should be ~1)")
    if not (0.8 < ry < 1.25 and 0.8 < rx < 1.25):
        print(f"[{BANNER}] WARNING: metric and cell positions disagree; "
              f"physical E and v_E magnitudes are suspect.")

    single_null = int(g["jyseps2_1"].values) == int(g["jyseps1_2"].values)
    ix1 = int(g["ixseps1"].values)
    ix2 = int(g["ixseps2"].values) if "ixseps2" in g else nx
    seps = sorted(i for i in (ix1, ix2) if 0 < i < nx)
    return dict(
        grid=g, rm=rm, zm=zm,
        ny_inner=ny_inner, bridge_cut=single_null, cuts=cuts, nbr=nbr,
        R=R, Z=Z, B=B, Bp=Bp, Bt=Bt, hthe=hthe, dx=dx, dy=dy,
        g11=g11, g23=g23, J=J,
        Cx=None if bxcvx is None else 2.0 * bxcvx / B,
        Cy=None if bxcvy is None else 2.0 * bxcvy / B,
        ix_sep=seps[0], ix_sep2=seps[1] if len(seps) > 1 else None,
    )


def radial_distance(geo, j):
    """Distance along the x-index line at column j, zero on the first separatrix."""
    R, Z, ix = geo["R"][:, j], geo["Z"][:, j], geo["ix_sep"]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(R), np.diff(Z)))])
    return s - 0.5 * (s[ix - 1] + s[ix])


# ══════════════════════════════════════════════════════════════════════════════
# Fields
# ══════════════════════════════════════════════════════════════════════════════

def _series(da):
    """(t, x, y) values with the toroidal dim dropped (MZ = 1)."""
    for d in ("zeta", "z"):
        if d in da.dims:
            da = da.isel({d: 0})
    return da.values


def _mass(ds, heat, s):
    return ME if s == "e" else heat._opt(ds, s, "AA", 2.0) * MP


def _velocity(src, heat, run, s):
    """Time-averaged parallel velocity [m/s]: V<s> if diagnosed, else NV/(m N)."""
    t0, t1 = run["t0"], run["t1"]
    if f"V{s}" in src.data_vars:
        return heat._tavg(src[f"V{s}"], t0, t1)
    N = src.get("Ne") if s == "e" else src.get(f"N{s}")
    if f"NV{s}" not in src.data_vars or N is None:
        return None
    n = heat._tavg(N, t0, t1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return heat._tavg(src[f"NV{s}"], t0, t1) / (_mass(run["ds"], heat, s) * n)


def _jpar(src, heat, run):
    """Parallel current density e(sum Z n_i V_i - n_e V_e) [A/m^2], or None."""
    t0, t1, ds = run["t0"], run["t1"], run["ds"]
    Ve = _velocity(src, heat, run, "e")
    if Ve is None or "Ne" not in src.data_vars:
        return None
    j = -heat._tavg(src["Ne"], t0, t1) * Ve
    for ion in run["ions"]:
        Vi = _velocity(src, heat, run, ion)
        if Vi is None or f"N{ion}" not in src.data_vars:
            return None
        Zi = heat._opt(ds, ion, "charge", 1.0)
        j = j + Zi * heat._tavg(src[f"N{ion}"], t0, t1) * Vi
    return QE * j


def _charged_pressure(src, heat, run):
    """Time-averaged sum of P_s [Pa] over the electrons and the ions, or None."""
    Ps = [heat._tavg(src[f"P{s}"], run["t0"], run["t1"]) for s in ["e"] + run["ions"]
          if f"P{s}" in src.data_vars]
    return sum(Ps) if Ps else None


def _divjdia(P, geo):
    """div(P Curlb_B) [A/m^3] in the form of BOUT's Div(Vector3D), d/dz = 0."""
    Fx = geo["J"] * P * geo["Cx"]
    Fy = geo["J"] * P * geo["Cy"]
    return (_ddx(Fx, geo["dx"]) + _ddy(Fy, geo["dy"], geo["nbr"])) / geo["J"]


def _anomalous_flow_xlow(n, D, geo):
    """
    Anomalous particle flow [s^-1] through the xlow face of each cell, as
    Div_a_Grad_perp_upwind_flows writes it (+ = increasing x). Cell 0 has none.
    """
    Jg = geo["J"] * geo["g11"]
    grad = (Jg[:-1] + Jg[1:]) * (n[1:] - n[:-1]) / (geo["dx"][:-1] + geo["dx"][1:])
    a = np.where(grad > 0, D[1:], D[:-1])
    flow = np.zeros_like(n, dtype=float)
    flow[1:] = -grad * a * geo["dy"][:-1] * 2 * np.pi
    return flow


def compute_fields(run, geo, heat):
    """
    Time-averaged (last --twindow steps) plasma and drift fields on the
    interior frame, full radial extent. Drift entries are None without phi.
    """
    ds, t0, t1 = run["ds"], run["t0"], run["t1"]
    ion = run["ions"][0] if run["ions"] else None
    avg = lambda v: heat._tavg(ds[v], t0, t1) if v in ds.data_vars else None  # noqa: E731

    f = dict(Ne=avg("Ne"), Te=avg("Te"), phi=avg("phi"), DivJdia=avg("DivJdia"),
             ion=ion, Ni=avg(f"N{ion}") if ion else None,
             Ti=avg(f"T{ion}") if ion else None)
    f["Vi"] = _velocity(ds, heat, run, ion) if ion else None
    f["Ve"] = _velocity(ds, heat, run, "e")
    f["jpar"] = _jpar(ds, heat, run)
    if ion and f["Vi"] is not None and f["Te"] is not None:
        Ti = f["Ti"] if f["Ti"] is not None else f["Te"]
        cs = np.sqrt(np.clip((f["Te"] + Ti) * QE / _mass(ds, heat, ion), 1e-30, None))
        f["Mach"] = f["Vi"] / cs

    f["Pch"] = _charged_pressure(ds, heat, run)
    if f["DivJdia"] is None and f["Pch"] is not None and geo["Cx"] is not None:
        f["DivJdia"] = _divjdia(f["Pch"], geo)
        print(f"[{BANNER}] {run['label']}: no DivJdia in the dump; reconstructed "
              f"from sum P_s * 2 bxcv / B.")

    if f["phi"] is None:
        print(f"[{BANNER}] {run['label']}: no phi in the dump -> no drift fields "
              f"(expected for a no-drift run).")
        return f

    phi = f["phi"]
    dpdx = _ddx(phi, geo["dx"])
    dpdy = _ddy(phi, geo["dy"], geo["nbr"])
    fac = geo["g11"] * geo["g23"] / geo["B"] ** 2
    vx = fac * dpdy                                      # contravariant v^x
    vy = -fac * dpdx                                     # contravariant v^y
    f["Er"] = -np.sqrt(geo["g11"]) * dpdx                # [V/m]
    f["Epol"] = -dpdy / geo["hthe"]                      # [V/m]
    f["vEr"] = vx / np.sqrt(geo["g11"])                  # [m/s]
    f["vEpol"] = vy * geo["hthe"]                        # [m/s]
    with np.errstate(divide="ignore", invalid="ignore"):
        f["phi_Te"] = phi / f["Te"]

    n = f["Ni"] if f["Ni"] is not None else f["Ne"]
    f["GammaEr"] = n * f["vEr"]                          # [m^-2 s^-1]
    # ExB particle flow through x faces, same form as the operator: n v^x J dy dz
    f["flowEx"] = n * vx * geo["J"] * geo["dy"] * 2 * np.pi     # [s^-1]
    pfx = avg(f"pf{ion}_tot_xlow") if ion else None
    origin = f"pf{ion}_tot_xlow"
    D = avg(f"anomalous_D_{ion}") if ion else None
    if pfx is None and D is not None:
        pfx = _anomalous_flow_xlow(n, D, geo)
        origin = f"anomalous_D_{ion} and N{ion} (reconstructed)"
    if pfx is None:
        print(f"[{BANNER}] {run['label']}: no pf{ion}_tot_xlow and no "
              f"anomalous_D_{ion} in the dump -> ExB/anomalous ratio map skipped.")
    elif not np.any(pfx[run["MXG"]:run["nx"] - run["MXG"]]):
        print(f"[{BANNER}] {run['label']}: anomalous radial particle flow is zero "
              f"everywhere (anomalous_D_{ion} = 0: neither [{ion}] anomalous_D nor "
              f"grid D_{ion} is set). ExB is the only cross-field ion transport; "
              f"ratio map skipped.")
    else:
        anom = np.full_like(pfx, np.nan, dtype=float)
        anom[:-1] = 0.5 * (pfx[:-1] + pfx[1:])
        f["flowAnom"] = anom
        with np.errstate(divide="ignore", invalid="ignore"):
            f["ratio"] = np.where(np.abs(anom) > 0,
                                  np.abs(f["flowEx"]) / np.abs(anom), np.nan)
        print(f"[{BANNER}] {run['label']}: anomalous radial flow from {origin}.")
    return f


# ══════════════════════════════════════════════════════════════════════════════
# Target data (guarded frame, as in the heat script)
# ══════════════════════════════════════════════════════════════════════════════

def target_data(run, geo, heat):
    """
    Per target slot: sheath-face profiles and the net current into the target.

    Face values are the average of the last domain cell and its guard cell in
    the y-guarded view (what sheath_boundary_simple sets). Falls back to
    cell-centre interior values if the guarded view is unusable.
    """
    fn, _ = heat._guard_map(run)
    if fn is not None and not heat._validate_guard_map(run, fn):
        fn = None
    src = run["dsg"] if fn is not None else run["ds"]
    t0, t1, sx = run["t0"], run["t1"], slice(run["MXG"], run["nx"] - run["MXG"])
    ion = run["ions"][0] if run["ions"] else None
    avg = lambda v: heat._tavg(src[v], t0, t1) if v in src.data_vars else None  # noqa: E731

    Ne, Te, phi = avg("Ne"), avg("Te"), avg("phi")
    jp = _jpar(src, heat, run)
    pf = avg(f"pf{ion}_tot_ylow") if ion else None
    ef = [a for a in (avg(f"ef{s}_tot_ylow") for s in run["species"]) if a is not None]
    Pch = _charged_pressure(src, heat, run)

    out = []
    for slot, tgt in enumerate(run["targets"]):
        if tgt is None:
            out.append(None)
            continue
        j, side = tgt
        if fn is not None:
            c = fn(j)
            gcol = c + 1 if side == "upper" else c - 1
            jf = heat._face_index(run, fn, j, side)
            face = lambda a: None if a is None else 0.5 * (a[:, c] + a[:, gcol])  # noqa: E731
        else:
            jf = j
            face = lambda a: None if a is None else a[:, j]                     # noqa: E731
        area = run["dA_y"][:, j]
        sgn = 1.0 if side == "upper" else -1.0          # + = into the target
        d = dict(slot=slot, name=heat.TARGET_SHORT[slot], theta=j, side=side,
                 r=radial_distance(geo, j), Ne=face(Ne), Te=face(Te), phi=face(phi))
        d["Gamma"] = None if pf is None else np.abs(pf[:, jf]) / area
        d["q"] = None if not ef else sum(np.abs(a[:, jf]) for a in ef) / area
        jf_face = face(jp)
        d["jin"] = None if jf_face is None else sgn * jf_face
        d["I"] = np.nan if jf_face is None else float(np.sum((d["jin"] * area)[sx]))
        if Pch is not None and geo["Cy"] is not None:
            Iy = sgn * face(Pch) * geo["Cy"][:, j] * geo["J"][:, j] * geo["dx"][:, j] * 2 * np.pi
            d["I_dia"] = float(np.sum(Iy[sx]))
        else:
            d["I_dia"] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            d["phi_Te"] = None if phi is None else d["phi"] / d["Te"]
        out.append(d)
    return out


def radial_dia_current(run, f, geo):
    """
    Net J_dia current [A] OUT of the domain through the inner (core + PF) and
    outer (wall) x boundaries, from the x-face average of J P Curlb_B^x dy dz.
    Returns (inner, outer) or None.
    """
    if f.get("Pch") is None or geo["Cx"] is None:
        return None
    flux = geo["J"] * f["Pch"] * geo["Cx"] * geo["dy"] * 2 * np.pi
    m, nx = run["MXG"], run["nx"]
    inner = -0.5 * (flux[m - 1] + flux[m])
    outer = 0.5 * (flux[nx - m - 1] + flux[nx - m])
    return float(np.nansum(inner)), float(np.nansum(outer))


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, args, name):
    path = os.path.join(args.outdir, f"{name}.{args.fmt}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[{BANNER}] Saved {path}")
    if not args.show:
        plt.close(fig)


def _map(pp, heat, run, geo, data, label, name, kind, args, title=None):
    """One polygon map. kind: 'seq', 'div' (symmetric RdBu_r), 'log', 'ratio'."""
    if data is None:
        print(f"[{BANNER}] Skipping map {name}: field not available.")
        return
    if geo["rm"] is None:
        print(f"[{BANNER}] Grid has no rm/zm corner mesh; cannot draw maps.")
        return
    interior = data[run["MXG"]:run["nx"] - run["MXG"]]
    log, cmap, vmin, vmax = False, "viridis", None, None
    if kind == "div":
        lo, hi = heat._clim(interior, args.clip)
        if lo is not None:
            m = max(abs(lo), abs(hi))
            vmin, vmax = -m, m
        cmap = "RdBu_r"
    elif kind == "log":
        log = True
    elif kind == "ratio":
        log, cmap, vmin, vmax = True, "PuOr_r", 1e-2, 1e2
    else:
        vmin, vmax = heat._clim(interior, args.clip)

    fig, ax = plt.subplots(figsize=(6, 10))
    pp.polygon_plot(ax, geo["rm"], geo["zm"], data, geo["ny_inner"], cmap=cmap,
                    vmin=vmin, vmax=vmax, colorbar_label=label, log=log,
                    bridge_cut=geo["bridge_cut"])
    _overlay(ax, pp, geo, args)
    ax.set_title(title or f"{name}   [{run['label']}]", fontsize=11)
    plt.tight_layout()
    _save(fig, args, name)


def _overlay(ax, pp, geo, args):
    """Separatrices and branch cuts on a polygon map, as --seps / --branch_cuts ask."""
    if args.seps:
        pp.plot_separatrices(ax, geo["rm"], geo["zm"], geo["grid"], geo["ny_inner"])
    if args.branch_cuts:
        pp.plot_branch_cuts(ax, geo["rm"], geo["zm"], geo["grid"], geo["ny_inner"])


def _map_pair(pp, heat, runs, geo, a, b, label, name, kind, args):
    """The same field for two runs side by side, on one shared colour scale."""
    if a is None or b is None:
        print(f"[{BANNER}] Skipping {name}: field missing in one of the runs.")
        return
    if geo["rm"] is None:
        print(f"[{BANNER}] Grid has no rm/zm corner mesh; cannot draw maps.")
        return
    m, nx = runs[0]["MXG"], runs[0]["nx"]
    both = np.concatenate([a[m:nx - m].ravel(), b[m:nx - m].ravel()])
    log = kind == "log"
    if log:
        pos = both[np.isfinite(both) & (both > 0)]
        vmin, vmax = (float(pos.min()), float(pos.max())) if pos.size else (None, None)
    else:
        vmin, vmax = heat._clim(both, args.clip)
    fig, axs = plt.subplots(1, 2, figsize=(11, 10))
    for ax, data, run in zip(axs, (a, b), runs):
        pp.polygon_plot(ax, geo["rm"], geo["zm"], data, geo["ny_inner"], cmap="viridis",
                        vmin=vmin, vmax=vmax, colorbar_label=label, log=log,
                        bridge_cut=geo["bridge_cut"])
        _overlay(ax, pp, geo, args)
        ax.set_title(run["label"], fontsize=11)
    plt.tight_layout()
    _save(fig, args, name)


def _mark_cuts(ax, geo):
    """Dotted lines at branch-cut interfaces, dashed at the ny_inner gap."""
    for cut in geo["cuts"]:
        ax.axvline(cut + 0.5, color="grey", ls=":", lw=0.8)
    ax.axvline(geo["ny_inner"] - 0.5, color="k", ls="--", lw=0.8)


def leg_entrances(run, geo, heat):
    """
    Poloidal column just past the branch cut on the leg side, for each target.

    A lower-side target (leg runs towards increasing y) enters at the last cell
    before the first cut at or above it; an upper-side target at the first cell
    after the last cut below it. The ny_inner target gap is not a leg cut.
    """
    gap = geo["ny_inner"] - 1
    cuts = [c for c in geo["cuts"] if c != gap]
    out = []
    for slot, tgt in enumerate(run["targets"]):
        if tgt is None:
            continue
        j, side = tgt
        if side == "lower":
            cand = [c for c in cuts if c >= j]
            col = min(cand) if cand else None
        else:
            cand = [c for c in cuts if c < j]
            col = max(cand) + 1 if cand else None
        if col is not None:
            out.append((f"{heat.TARGET_SHORT[slot]} leg entrance", col))
    return out


def _mark_seps(ax, r, geo):
    """Dashed vertical lines at the separatrices on a radial-distance axis."""
    ax.axvline(0.0, color="k", ls="--", lw=0.8)
    ix2 = geo["ix_sep2"]
    if ix2 is not None:
        ax.axvline(1e3 * 0.5 * (r[ix2 - 1] + r[ix2]), color="grey", ls=":", lw=0.8)


# ══════════════════════════════════════════════════════════════════════════════
# Tier 0 — convergence
# ══════════════════════════════════════════════════════════════════════════════

def tier0(runs, geo, heat, args):
    run = runs[0]
    ds, sx = run["ds"], slice(run["MXG"], run["nx"] - run["MXG"])
    t = np.asarray(ds["t"].values, dtype=float)
    fig, axs = plt.subplots(3, 2, figsize=(12, 11))
    ax = axs.ravel()

    ix, ix_far = geo["ix_sep"], run["nx"] - run["MXG"] - 2
    j_omp = int(np.argmax(geo["R"][ix]))
    probes = {f"OMP sep (x={ix})": (ix, j_omp), f"OMP far SOL (x={ix_far})": (ix_far, j_omp)}
    for slot, tgt in enumerate(run["targets"]):
        if tgt is not None:
            probes[f"{heat.TARGET_SHORT[slot]} (x={ix})"] = (ix, tgt[0])

    if "phi" in ds.data_vars:
        phi_t = _series(ds["phi"])
        for lab, (i, j) in probes.items():
            ax[0].plot(t * 1e3, phi_t[:, i, j], label=lab)
        ax[0].set_ylabel(r"$\phi$ [V]")
        ax[0].legend(fontsize="small")
        if len(t) > 1:
            dt = np.diff(t)[:, None, None]
            rate = np.abs(np.diff(phi_t[:, sx], axis=0)) / dt
            tm = 0.5 * (t[1:] + t[:-1]) * 1e3
            ax[1].semilogy(tm, np.nanmax(rate, axis=(1, 2)), label="max")
            ax[1].semilogy(tm, np.sqrt(np.nanmean(rate ** 2, axis=(1, 2))), label="rms")
            ax[1].legend(fontsize="small")
            print(f"[{BANNER}] Last-step |dphi/dt|: max {np.nanmax(rate[-1]):.3e} V/s, "
                  f"rms {np.sqrt(np.nanmean(rate[-1] ** 2)):.3e} V/s")
        ax[1].set_ylabel(r"|d$\phi$/dt| [V/s]")
    else:
        ax[0].text(0.5, 0.5, "no phi in dump", ha="center", transform=ax[0].transAxes)

    # Target flows vs time (guarded frame for the correct upper-target faces).
    fn, _ = heat._guard_map(run)
    if fn is not None and not heat._validate_guard_map(run, fn):
        fn = None
    src = run["dsg"] if fn is not None else ds
    ion = run["ions"][0] if run["ions"] else None
    pf = _series(src[f"pf{ion}_tot_ylow"]) if ion and f"pf{ion}_tot_ylow" in src else None
    ef = [_series(src[f"ef{s}_tot_ylow"]) for s in run["species"]
          if f"ef{s}_tot_ylow" in src.data_vars]
    for slot, tgt in enumerate(run["targets"]):
        if tgt is None:
            continue
        jf = heat._face_index(run, fn, *tgt) if fn is not None else tgt[0]
        lab = heat.TARGET_SHORT[slot]
        if pf is not None:
            ax[2].plot(t * 1e3, np.sum(np.abs(pf[:, sx, jf]), axis=1), label=lab)
        if ef:
            ax[3].plot(t * 1e3, 1e-6 * sum(np.sum(np.abs(a[:, sx, jf]), axis=1) for a in ef),
                       label=lab)
    ax[2].set_ylabel(f"{ion} flow to target [s$^{{-1}}$]")
    ax[3].set_ylabel("plasma power to target [MW]")
    for a in ax[2:4]:
        a.legend(fontsize="small")

    dV = run["dV"][sx]
    if "Ne" in ds.data_vars:
        ax[4].plot(t * 1e3, np.nansum(_series(ds["Ne"])[:, sx] * dV, axis=(1, 2)))
        ax[4].set_ylabel("total electrons")
    Ps = [s for s in ["e"] + run["ions"] if f"P{s}" in ds.data_vars]
    if Ps:
        E = sum(np.nansum(1.5 * _series(ds[f"P{s}"])[:, sx] * dV, axis=(1, 2)) for s in Ps)
        ax[5].plot(t * 1e3, E)
        ax[5].set_ylabel(f"plasma thermal energy [J]  ({'+'.join(Ps)})")
    for a in ax:
        a.set_xlabel("t [ms]")
        a.grid(alpha=0.3)
    fig.suptitle(f"Tier 0 — convergence   [{run['label']}]")
    plt.tight_layout()
    _save(fig, args, "T0_convergence")


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 — maps
# ══════════════════════════════════════════════════════════════════════════════

def tier1(runs, fields, geo, pp, heat, args):
    run, f = runs[0], fields[0]
    s = lambda k, fac=1.0: None if f.get(k) is None else fac * f[k]   # noqa: E731
    ion = f["ion"]
    maps = [
        ("T1_phi", s("phi"), r"$\phi$ [V]", "div"),
        ("T1_phi_over_Te", s("phi_Te"), r"$\phi / T_e$", "div"),
        ("T1_Er", s("Er", 1e-3), r"$E_r$ [kV/m]  (+ = increasing x)", "div"),
        ("T1_Epol", s("Epol", 1e-3), r"$E_{pol}$ [kV/m]  (+ = increasing y)", "div"),
        ("T1_vE_radial", s("vEr"), r"$v_{E,r}$ [m/s]  (+ = increasing x)", "div"),
        ("T1_vE_poloidal", s("vEpol"), r"$v_{E,pol}$ [m/s]  (+ = increasing y)", "div"),
        ("T1_GammaE_radial", s("GammaEr"),
         rf"$n_{{{ion}}} v_{{E,r}}$ [m$^{{-2}}$ s$^{{-1}}$]", "div"),
        ("T1_ratio_ExB_anom", s("ratio"),
         r"$|\Gamma_{E\times B,x}| / |\Gamma_{anom,x}|$", "ratio"),
        ("T1_jpar", s("jpar", 1e-3), r"$j_\parallel$ [kA/m$^2$]  (+ = along +y)", "div"),
        ("T1_DivJdia", s("DivJdia"), r"$\nabla\cdot J_{dia}$ [A/m$^3$]", "div"),
        ("T1_Ne", s("Ne"), r"$n_e$ [m$^{-3}$]", "log"),
        ("T1_Te", s("Te"), r"$T_e$ [eV]", "log"),
    ]
    for name, data, label, kind in maps:
        _map(pp, heat, run, geo, data, label, name, kind, args)


# Directional drift quantities, plotted as 1D cuts rather than maps.
CUT_PANELS = [
    ("Er", r"$E_r$ [kV/m]  (+ = increasing x)", 1e-3),
    ("vEpol", r"$v_{E,pol}$ [m/s]  (+ = increasing y)", 1.0),
    ("Epol", r"$E_{pol}$ [kV/m]  (+ = increasing y)", 1e-3),
    ("vEr", r"$v_{E,r}$ [m/s]  (+ = increasing x)", 1.0),
    ("GammaEr", r"$n_i v_{E,r}$ [m$^{-2}$ s$^{-1}$]", 1.0),
]
RUN_STYLES = ["-", "--", ":", "-."]


def tier1_radial_cuts(runs, fields, geo, heat, args):
    """Components vs r - r_sep at the OMP and at each leg entrance."""
    ix = geo["ix_sep"]
    locs = [("OMP", int(np.argmax(geo["R"][ix])))] + leg_entrances(runs[0], geo, heat)
    sx = slice(runs[0]["MXG"], runs[0]["nx"] - runs[0]["MXG"])
    fig, axs = plt.subplots(len(CUT_PANELS), 1, figsize=(10, 2.6 * len(CUT_PANELS)),
                            sharex=True)
    for (key, lab, fac), ax in zip(CUT_PANELS, axs):
        for k, (run, f) in enumerate(zip(runs, fields)):
            if f.get(key) is None:
                continue
            for n, (name, j) in enumerate(locs):
                r = radial_distance(geo, j)
                ax.plot(1e3 * r[sx], fac * f[key][sx, j],
                        color=RUN_COLORS[n % len(RUN_COLORS)] if n < 4 else "tab:purple",
                        ls=RUN_STYLES[k % len(RUN_STYLES)],
                        label=f"{name} (j={j})" + (f"  [{run['label']}]" if len(runs) > 1 else ""))
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize="x-small", ncol=2)
    axs[-1].set_xlabel(r"$r - r_{sep}$ [mm]")
    fig.suptitle(f"Tier 1 — radial cuts of the drift components  [{runs[0]['label']}]")
    plt.tight_layout()
    _save(fig, args, "T1_radial_cuts")


def tier1_poloidal_cuts(runs, fields, geo, args):
    """Components vs poloidal index on the first SOL ring and last ring inside."""
    ix = geo["ix_sep"]
    rings = [(f"x={ix} (first outside separatrix)", ix),
             (f"x={ix - 1} (last inside separatrix)", ix - 1)]
    fig, axs = plt.subplots(len(CUT_PANELS), 1, figsize=(11, 2.4 * len(CUT_PANELS)),
                            sharex=True)
    for (key, lab, fac), ax in zip(CUT_PANELS, axs):
        for k, (run, f) in enumerate(zip(runs, fields)):
            if f.get(key) is None:
                continue
            for n, (name, i) in enumerate(rings):
                ax.plot(fac * f[key][i], color=RUN_COLORS[n], ls=RUN_STYLES[k % len(RUN_STYLES)],
                        label=name + (f"  [{run['label']}]" if len(runs) > 1 else ""))
        _mark_cuts(ax, geo)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize="x-small")
    axs[-1].set_xlabel("poloidal index  (dotted: branch cuts, dashed: ny_inner gap)")
    fig.suptitle(f"Tier 1 — poloidal cuts of the drift components  [{runs[0]['label']}]")
    plt.tight_layout()
    _save(fig, args, "T1_poloidal_cuts")


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2 — profiles
# ══════════════════════════════════════════════════════════════════════════════

def tier2_omp(runs, fields, geo, args):
    ix = geo["ix_sep"]
    j = int(np.argmax(geo["R"][ix]))
    r = radial_distance(geo, j)
    sx = slice(runs[0]["MXG"], runs[0]["nx"] - runs[0]["MXG"])
    panels = [("Ne", r"$n_e$ [m$^{-3}$]", 1), ("Te", r"$T_e$, $T_i$ (dashed) [eV]", 1),
              ("phi", r"$\phi$ [V]", 1), ("Er", r"$E_r$ [kV/m]", 1e-3),
              ("vEpol", r"$v_{E,pol}$ [m/s]", 1), ("phi_Te", r"$\phi/T_e$", 1)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for (key, lab, fac), ax in zip(panels, axs.ravel()):
        for k, (run, f) in enumerate(zip(runs, fields)):
            if f.get(key) is None:
                continue
            c = RUN_COLORS[k % len(RUN_COLORS)]
            ax.plot(1e3 * r[sx], fac * f[key][sx, j], color=c, label=run["label"])
            if key == "Te" and f.get("Ti") is not None:
                ax.plot(1e3 * r[sx], f["Ti"][sx, j], color=c, ls="--")
        _mark_seps(ax, r, geo)
        ax.set_ylabel(lab)
        ax.set_xlabel(r"$r - r_{sep}$ [mm]")
        ax.grid(alpha=0.3)
        if key == "Ne":
            ax.set_yscale("log")
        ax.legend(fontsize="small")
    fig.suptitle(f"Tier 2 — radial profiles at the OMP (theta index {j})")
    plt.tight_layout()
    _save(fig, args, "T2_omp_radial")


def tier2_targets(runs, tdata, geo, args):
    n_t = sum(t is not None for t in tdata[0])
    cols = [("Ne", r"$n_e$ [m$^{-3}$]", 1), ("Te", r"$T_e$ [eV]", 1),
            ("Gamma", r"$\Gamma_\parallel$ [m$^{-2}$s$^{-1}$]", 1),
            ("q", r"$q_\parallel$ [MW/m$^2$]", 1e-6), ("phi", r"$\phi$ [V]", 1),
            ("phi_Te", r"$\phi/T_e$", 1),
            ("jin", r"$j_\parallel$ into target [kA/m$^2$]", 1e-3)]
    fig, axs = plt.subplots(n_t, len(cols), figsize=(3.2 * len(cols), 2.8 * n_t),
                            squeeze=False)
    sx = slice(runs[0]["MXG"], runs[0]["nx"] - runs[0]["MXG"])
    row = 0
    for slot, t0d in enumerate(tdata[0]):
        if t0d is None:
            continue
        for c, (key, lab, fac) in enumerate(cols):
            ax = axs[row, c]
            for k, (run, td) in enumerate(zip(runs, tdata)):
                d = td[slot]
                if d is None or d.get(key) is None:
                    continue
                ax.plot(1e3 * d["r"][sx], fac * d[key][sx],
                        color=RUN_COLORS[k % len(RUN_COLORS)], label=run["label"])
            _mark_seps(ax, t0d["r"], geo)
            if key == "jin":
                ax.axhline(0, color="k", lw=0.5)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(lab, fontsize=10)
            if row == n_t - 1:
                ax.set_xlabel(r"$r - r_{sep}$ along target [mm]")
            if c == 0:
                ax.set_ylabel(t0d["name"])
                ax.legend(fontsize="x-small")
        row += 1
    fig.suptitle("Tier 2 — sheath-face profiles at each strike point")
    plt.tight_layout()
    _save(fig, args, "T2_targets")


def tier2_ring(runs, fields, geo, args):
    ix = args.ring if args.ring is not None else geo["ix_sep"]
    panels = [("Ne", r"$n_e$ [m$^{-3}$]"), ("Te", r"$T_e$, $T_i$ (dashed) [eV]"),
              ("phi", r"$\phi$ [V]"), ("Vi", r"$V_{\parallel i}$, $V_{\parallel e}$ (dashed) [m/s]"),
              ("jpar", r"$j_\parallel$ [kA/m$^2$]"), ("Mach", r"$M_\parallel = V_i/c_s$")]
    fig, axs = plt.subplots(len(panels), 1, figsize=(11, 2.2 * len(panels)), sharex=True)
    for (key, lab), ax in zip(panels, axs):
        for k, (run, f) in enumerate(zip(runs, fields)):
            if f.get(key) is None:
                continue
            c = RUN_COLORS[k % len(RUN_COLORS)]
            fac = 1e-3 if key == "jpar" else 1.0
            ax.plot(fac * f[key][ix], color=c, label=run["label"])
            if key == "Te" and f.get("Ti") is not None:
                ax.plot(f["Ti"][ix], color=c, ls="--")
            if key == "Vi" and f.get("Ve") is not None:
                ax.plot(f["Ve"][ix], color=c, ls="--")
        _mark_cuts(ax, geo)
        if key in ("jpar", "Vi", "Mach", "phi"):
            ax.axhline(0, color="k", lw=0.5)
        if key == "Ne":
            ax.set_yscale("log")
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize="small")
    axs[-1].set_xlabel("poloidal index  (dotted: branch cuts, dashed: ny_inner gap)")
    fig.suptitle(f"Tier 2 — along ring x = {ix}  "
                 f"(first separatrix at x = {geo['ix_sep']})")
    plt.tight_layout()
    _save(fig, args, f"T2_ring_x{ix}")


# ══════════════════════════════════════════════════════════════════════════════
# Tier 3 — per-strike-point budget
# ══════════════════════════════════════════════════════════════════════════════

def tier3(runs, fields, tdata, geo, heat, args):
    """
    Power, particle flow and the current budget per strike point. Returns one
    dict per run (case, family, P and F per target slot) for the all-case summary.
    """
    m1s = [heat.method1_energy_flow(run) for run in runs]
    slots = [s for s, t in enumerate(runs[0]["targets"]) if t is not None]
    names = [heat.TARGET_SHORT[s] for s in slots]
    x = np.arange(len(slots))
    xc = np.arange(len(slots) + 2)
    w = 0.8 / len(runs)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8),
                            gridspec_kw=dict(width_ratios=[1, 1, 1.4]))

    print(f"\n[{BANNER}] Per-strike-point budget (avg over last {args.twindow} steps)")
    lw = max(20, max(len(r["label"]) for r in runs) + 2)
    print(f"  {'run':<{lw}}{'target':<11}{'P [MW]':>10}{'flow [s^-1]':>14}"
          f"{'I_par [A]':>12}{'I_dia [A]':>12}")
    budgets = []
    for k, (run, f, m1, td) in enumerate(zip(runs, fields, m1s, tdata)):
        drift = f.get("phi") is not None
        P, F, Ip, Id = [], [], [], []
        for s in slots:
            pt = m1["per_target"][s]
            P.append(1e-6 * pt["power"] if pt else np.nan)
            F.append(pt["flux"] if pt else np.nan)
            Ip.append(td[s]["I"] if td[s] else np.nan)
            Id.append(td[s]["I_dia"] if (td[s] and drift) else np.nan)
            print(f"  {run['label']:<{lw}}{heat.TARGET_SHORT[s]:<11}{P[-1]:>10.4f}"
                  f"{F[-1]:>14.3e}{Ip[-1]:>12.2f}{Id[-1]:>12.2f}")
        rad = radial_dia_current(run, f, geo) if drift else None
        print(f"  {run['label']:<{lw}}{'SUM':<11}{np.nansum(P):>10.4f}"
              f"{np.nansum(F):>14.3e}{np.nansum(Ip):>12.2f}{np.nansum(Id):>12.2f}")
        if rad is not None:
            total = np.nansum(Ip) + np.nansum(Id) + rad[0] + rad[1]
            print(f"  {run['label']:<{lw}}J_dia out through inner x boundary "
                  f"{rad[0]:>10.2f} A, outer x boundary {rad[1]:>10.2f} A")
            print(f"  {run['label']:<{lw}}NET current out of the domain "
                  f"{total:>10.2f} A   <- should be ~0  "
                  f"(scale: sum |I_par| = {np.nansum(np.abs(Ip)):.2f} A)")
        else:
            total = np.nansum(Ip)
            print(f"  {run['label']:<{lw}}no phi -> no charge balance in this "
                  f"model; J_dia not counted")

        c = RUN_COLORS[k % len(RUN_COLORS)]
        off = (k - (len(runs) - 1) / 2) * w
        axs[0].bar(x + off, P, w, color=c, label=run["label"])
        axs[1].bar(x + off, F, w, color=c, label=run["label"])
        Ipc = np.array(Ip + [0.0, 0.0])
        Idc = np.nan_to_num(np.array(Id + (list(rad) if rad else [np.nan, np.nan])))
        axs[2].bar(xc + off, Ipc, w, color=c,
                   label=f"{run['label']}  (net {total:.1f} A)")
        axs[2].bar(xc + off, Idc, w, bottom=Ipc, color=c, alpha=0.45, hatch="//",
                   edgecolor="k", lw=0.3)
        budgets.append(dict(case=run["case"], family=run["family"],
                            P=dict(zip(slots, P)), F=dict(zip(slots, F))))
    for ax, lab, ticks, tl in ((axs[0], "plasma power to target [MW]", x, names),
                               (axs[1], "ion flow to target [s$^{-1}$]", x, names),
                               (axs[2], "current out of the domain [A]", xc,
                                names + ["inner x\nbdry", "outer x\nbdry"])):
        ax.set_xticks(ticks)
        ax.set_xticklabels(tl, fontsize=9)
        ax.set_ylabel(lab)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize="x-small")
    h, lab = axs[2].get_legend_handles_labels()
    axs[2].legend(h + [Patch(facecolor="w", hatch="//", edgecolor="k")],
                  lab + [r"hatched: $J_{dia}$ part"], fontsize="x-small")
    fig.suptitle(f"Tier 3 — per strike point   [{runs[0]['case']}]")
    plt.tight_layout()
    _save(fig, args, "T3_strike_points")

    run = runs[0]
    ion = run["ions"][0] if run["ions"] else None
    if ion:
        m2 = dict(ion=ion, AA=heat._opt(run["ds"], ion, "AA", 2.0),
                  Z=heat._opt(run["ds"], ion, "charge", 1.0))
        rows = heat.sheath_potential_check(run, m2)
        if rows:
            print(f"\n[{BANNER}] Sheath potential check [{run['label']}] "
                  f"(n-weighted over target):")
            print(f"  {'target':<11}{'Te [eV]':>9}{'phi [V]':>9}{'phi/Te':>9}"
                  f"{'ambipolar':>11}{'ve/cs':>8}")
            for s, r in enumerate(rows):
                if r:
                    print(f"  {heat.TARGET_SHORT[s]:<11}{r['Te']:>9.2f}{r['phi']:>9.2f}"
                          f"{r['ratio']:>9.2f}{r['amb']:>11.2f}{r['ve_cs']:>8.2f}")
    return budgets


def summary_all_cases(budgets, heat, args):
    """Power and particle flow per strike point for every case, one bar per run family."""
    cases = [b[0]["case"] for b in budgets]
    families = list(dict.fromkeys(r["family"] for b in budgets for r in b))
    slots = sorted({s for b in budgets for r in b for s in r["P"]})
    rows = slots + ["total"]
    x = np.arange(len(cases))
    w = 0.8 / len(families)
    fig, axs = plt.subplots(len(rows), 2, figsize=(max(10, 1.4 * len(cases) + 4),
                                                   2.4 * len(rows)),
                            sharex=True, squeeze=False)

    def value(run, key, slot):
        if run is None:
            return np.nan
        if slot == "total":
            return np.nansum(list(run[key].values()))
        return run[key].get(slot, np.nan)

    print(f"\n[{BANNER}] All-case summary: power [MW] / ion flow [s^-1] per strike point")
    print(f"  {'case':<12}{'run':<12}" + "".join(
        f"{(heat.TARGET_SHORT[s] if s != 'total' else 'total'):>24}" for s in rows))
    for k, fam in enumerate(families):
        off = (k - (len(families) - 1) / 2) * w
        picked = [next((r for r in b if r["family"] == fam), None) for b in budgets]
        for r, slot in enumerate(rows):
            for col, key, fac in ((0, "P", 1.0), (1, "F", 1.0)):
                axs[r, col].bar(x + off, [fac * value(p, key, slot) for p in picked], w,
                                color=RUN_COLORS[k % len(RUN_COLORS)], label=fam)
        for case, p in zip(cases, picked):
            if p is None:
                continue
            print(f"  {case:<12}{fam:<12}" + "".join(
                f"{value(p, 'P', s):>11.4f} /{value(p, 'F', s):>11.3e}" for s in rows))
    for r, slot in enumerate(rows):
        name = heat.TARGET_SHORT[slot] if slot != "total" else "all targets"
        axs[r, 0].set_ylabel(f"{name}\nP [MW]", fontsize=9)
        axs[r, 1].set_ylabel(f"{name}\nion flow [s$^{{-1}}$]", fontsize=9)
        for ax in axs[r]:
            ax.grid(alpha=0.3, axis="y")
    axs[0, 0].legend(fontsize="small")
    for ax in axs[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(cases)
    fig.suptitle("Tier 3 — power and particle flow per strike point, all cases\n"
                 "(SN fills SP1 and SP2 only: its two legs at the poloidal ends)")
    plt.tight_layout()
    _save(fig, args, "T3_all_cases")


# ══════════════════════════════════════════════════════════════════════════════
# Tier 4 — drifts vs no drifts
# ══════════════════════════════════════════════════════════════════════════════

DELTA_MAPS = [
    ("Te", r"$\Delta T_e$ [eV]", "T4_dTe", False, 1.0),
    ("Ti", r"$\Delta T_i$ [eV]", "T4_dTi", False, 1.0),
    ("Ne", r"$\Delta n_e / n_e$", "T4_dNe_rel", True, 1.0),
    ("Mach", r"$\Delta M_\parallel$", "T4_dMach", False, 1.0),
    ("jpar", r"$\Delta j_\parallel$ [kA/m$^2$]", "T4_djpar", False, 1e-3),
    ("phi", r"$\Delta \phi$ [V]", "T4_dphi", False, 1.0),
]


def tier4(runs, fields, tdata, geo, pp, heat, args):
    if len(runs) < 2:
        print(f"[{BANNER}] Tier 4 needs --compare; skipping.")
        return
    a, b = fields[0], fields[1]
    lab = f"{runs[0]['family']} - {runs[1]['family']}"
    _map_pair(pp, heat, runs, geo, a.get("Ne"), b.get("Ne"), r"$n_e$ [m$^{-3}$]",
              "T4_Ne_side_by_side", "log", args)
    _map_pair(pp, heat, runs, geo, a.get("Te"), b.get("Te"), r"$T_e$ [eV]",
              "T4_Te_side_by_side", "log", args)
    for key, sym, name, rel, fac in DELTA_MAPS:
        if a.get(key) is None or b.get(key) is None:
            print(f"[{BANNER}] Skipping {name}: {key} missing in one of the runs.")
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (a[key] - b[key]) / b[key] if rel else fac * (a[key] - b[key])
        _map(pp, heat, runs[0], geo, d, f"{sym}  ({lab})", name, "div", args,
             title=f"{sym}   [{runs[0]['case']}]")
    tier4_targets(runs, tdata, geo, args)


def tier4_targets(runs, tdata, geo, args):
    """Run 1 minus run 2 for the sheath-face profiles at each strike point."""
    cols = [("Ne", r"$\Delta n_e$ [m$^{-3}$]", 1), ("Te", r"$\Delta T_e$ [eV]", 1),
            ("Gamma", r"$\Delta \Gamma_\parallel$ [m$^{-2}$s$^{-1}$]", 1),
            ("q", r"$\Delta q_\parallel$ [MW/m$^2$]", 1e-6)]
    a, b = tdata[0], tdata[1]
    slots = [s for s, d in enumerate(a) if d is not None and b[s] is not None]
    if not slots:
        return
    sx = slice(runs[0]["MXG"], runs[0]["nx"] - runs[0]["MXG"])
    fig, axs = plt.subplots(len(slots), len(cols), figsize=(3.4 * len(cols), 2.8 * len(slots)),
                            squeeze=False)
    for row, s in enumerate(slots):
        for c, (key, lab, fac) in enumerate(cols):
            ax = axs[row, c]
            if a[s].get(key) is not None and b[s].get(key) is not None:
                ax.plot(1e3 * a[s]["r"][sx], fac * (a[s][key] - b[s][key])[sx], color="k")
            _mark_seps(ax, a[s]["r"], geo)
            ax.axhline(0, color="k", lw=0.5)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(lab, fontsize=10)
            if row == len(slots) - 1:
                ax.set_xlabel(r"$r - r_{sep}$ along target [mm]")
            if c == 0:
                ax.set_ylabel(a[s]["name"])
    fig.suptitle(f"Tier 4 — {runs[0]['family']} minus {runs[1]['family']} at each "
                 f"strike point   [{runs[0]['case']}]")
    plt.tight_layout()
    _save(fig, args, "T4_targets_delta")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _str2bool(value):
    """Parse a CLI boolean so ``--drifts False`` works."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _has_drifts(run):
    """True if the run computes phi: relax_potential or vorticity in [hermes] components."""
    try:
        comps = str(run["ds"].attrs["options"]["hermes"]["components"])
    except Exception:
        comps = ""
    return ("relax_potential" in comps or "vorticity" in comps
            or "phi" in run["ds"].data_vars)


def process_case(case, other, pp, heat, args, tiers):
    """Load one case (and its --compare match), run the tiers. Returns Tier 3 budgets."""
    runs = []
    for c, fam in ((case, args.label), (other, args.compare_label)):
        if c is None:
            continue
        run = heat.load_run(c["path"], c["grid"], twindow=args.twindow)
        run.update(case=c["label"], family=fam, label=f"{c['label']} {fam}")
        run["drifts"] = _has_drifts(run)
        runs.append(run)

    if runs[0]["drifts"] != args.drifts:
        print(f"[{BANNER}] WARNING: --drifts {args.drifts} but {case['label']} "
              f"{'has' if runs[0]['drifts'] else 'has no'} relax_potential/vorticity "
              f"in BOUT.settings. Figures still go to {args.outdir}.")
    if len(runs) > 1:
        if runs[1]["drifts"] == runs[0]["drifts"]:
            print(f"[{BANNER}] WARNING: the --compare run has the same drift setting "
                  f"as the main run; Tier 4 is then not drifts vs no drifts.")
        if (runs[1]["nx"], runs[1]["ny"]) != (runs[0]["nx"], runs[0]["ny"]):
            print(f"[{BANNER}] --compare run for {case['label']} is on a different "
                  f"mesh; dropping it.")
            runs = runs[:1]

    geo = load_geometry(case["grid"], runs[0])
    seps = [geo["ix_sep"]] + ([geo["ix_sep2"]] if geo["ix_sep2"] is not None else [])
    print(f"[{BANNER}] Separatrices at x = {seps}; "
          f"y branch-cut interfaces after j = {geo['cuts']}")
    fields = [compute_fields(run, geo, heat) for run in runs]
    tdata = [target_data(run, geo, heat) for run in runs]

    jobs = [
        (0, "Tier 0", lambda: tier0(runs, geo, heat, args)),
        (1, "Tier 1 maps", lambda: tier1(runs, fields, geo, pp, heat, args)),
        (1, "Tier 1 radial cuts", lambda: tier1_radial_cuts(runs, fields, geo, heat, args)),
        (1, "Tier 1 poloidal cuts", lambda: tier1_poloidal_cuts(runs, fields, geo, args)),
        (2, "Tier 2 OMP", lambda: tier2_omp(runs, fields, geo, args)),
        (2, "Tier 2 targets", lambda: tier2_targets(runs, tdata, geo, args)),
        (2, "Tier 2 ring", lambda: tier2_ring(runs, fields, geo, args)),
        (3, "Tier 3", lambda: tier3(runs, fields, tdata, geo, heat, args)),
        (4, "Tier 4", lambda: tier4(runs, fields, tdata, geo, pp, heat, args)),
    ]
    budgets = None
    for tier, name, job in jobs:
        if tier not in tiers:
            continue
        try:
            res = job()
            if tier == 3:
                budgets = res
        except Exception:                      # one broken plot must not kill the rest
            print(f"[{BANNER}] {name} failed:")
            traceback.print_exc()
    return budgets


def main():
    parser = argparse.ArgumentParser(
        description="First-look drift plots for Hermes-3 relax_potential runs "
                    "(CLAUDE_DRIFTS.md §10 and Appendix A).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("sims_dir",
                        help="Folder containing the SF<angle>/SN simulation folders")
    parser.add_argument("grid_dir",
                        help="Folder containing the grid files (*sf<angle>*.nc, *lsn*.nc)")
    parser.add_argument("--only", nargs="+", default=None, metavar="CASE",
                        help="Only load these sim folders, e.g. --only SF45 SF165")
    parser.add_argument("--drifts", type=_str2bool, default=True, metavar="True/False",
                        help="Are the sims_dir runs drift runs? Picks <outdir>/Drifts "
                             "or <outdir>/No_drifts, and is checked against BOUT.settings")
    parser.add_argument("--compare", default=None,
                        help="Second sims_dir (e.g. the matched no-drift runs); "
                             "each case is paired with the folder of the same name")
    parser.add_argument("--label", default=None,
                        help="Label for the sims_dir runs (default: drifts / no drifts)")
    parser.add_argument("--compare_label", default=None,
                        help="Label for the --compare runs (default: the opposite)")
    parser.add_argument("--tiers", default="0,1,2,3,4",
                        help="Comma-separated tiers to produce")
    parser.add_argument("--twindow", type=int, default=5,
                        help="Average maps/profiles over the last N timesteps")
    parser.add_argument("--ring", type=int, default=None,
                        help="x index for the along-ring profile "
                             "(default: first cell outside the first separatrix)")
    parser.add_argument("--clip", type=float, default=99.5,
                        help="Percentile for colour limits (100 = true extrema)")
    parser.add_argument("--seps", type=_str2bool, default=True, metavar="True/False",
                        help="Overlay the separatrices on the maps")
    parser.add_argument("--branch_cuts", type=_str2bool, default=True, metavar="True/False",
                        help="Overlay the branch cuts on the maps")
    parser.add_argument("--outdir",
                        default="/Users/sruiz/Dev/PhD/SF_Sims/Modified_SF_Runs/"
                                "Post_Processing",
                        help="Figures go to <outdir>/<Drifts|No_drifts>/<case>/")
    parser.add_argument("--fmt", default="png", help="Figure format")
    parser.add_argument("--show", action="store_true",
                        help="Also show the figures interactively")
    parser.add_argument("--pp_path", default=None,
                        help="Explicit path to Hermes-3_postprocessing.py")
    parser.add_argument("--heat_path", default=None,
                        help="Explicit path to hermes-heat-pp-method-comparison.py")
    parser.add_argument("--sf_path", default=None,
                        help="Explicit path to SF_heat_transport_pp_xhermes_withfluxes.py")
    args = parser.parse_args()

    for d in filter(None, (args.sims_dir, args.grid_dir, args.compare)):
        if not os.path.isdir(d):
            sys.exit(f"Not a directory: {d}")
    tiers = {int(t) for t in args.tiers.split(",") if t.strip()}
    args.label = args.label or ("drifts" if args.drifts else "no drifts")
    args.compare_label = args.compare_label or ("no drifts" if args.drifts else "drifts")
    root = os.path.join(args.outdir, "Drifts" if args.drifts else "No_drifts")
    os.makedirs(root, exist_ok=True)

    pp = _load_module("hermes3_pp", _PP_CANDIDATES, args.pp_path)
    heat = _load_module("hermes_heat_pp", _HEAT_CANDIDATES, args.heat_path)
    _use_aligned_open(heat, pp)
    sf = heat._load_sf(args.sf_path)

    cases = sf.discover_cases(args.sims_dir, args.grid_dir, only=args.only)
    if not cases:
        sys.exit("No simulations found.")
    matched = {}
    if args.compare:
        matched = {c["label"]: c for c in
                   sf.discover_cases(args.compare, args.grid_dir, only=args.only)}

    budgets = []
    for case in cases:
        out = os.path.join(root, case["label"])
        os.makedirs(out, exist_ok=True)
        cargs = argparse.Namespace(**{**vars(args), "outdir": out})
        print(f"\n[{BANNER}] ══════ {case['label']}  ->  {out}")
        other = matched.get(case["label"])
        if args.compare and other is None:
            print(f"[{BANNER}] No {case['label']} in {args.compare}; Tier 4 skipped.")
        try:
            b = process_case(case, other, pp, heat, cargs, tiers)
        except Exception:
            print(f"[{BANNER}] {case['label']} failed:")
            traceback.print_exc()
            continue
        if b:
            budgets.append(b)

    if 3 in tiers and budgets:
        try:
            summary_all_cases(budgets, heat, argparse.Namespace(**{**vars(args),
                                                                   "outdir": root}))
        except Exception:
            print(f"[{BANNER}] All-case summary failed:")
            traceback.print_exc()

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
