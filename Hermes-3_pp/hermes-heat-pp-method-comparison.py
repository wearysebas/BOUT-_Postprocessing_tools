#!/usr/bin/env python3
"""
hermes-heat-pp.py — power-balance audit for Hermes-3 runs.

Answers one question: **does the power that goes in come back out?**  The same
number is computed four independent ways and printed side by side, so a leak
shows up as a disagreement between them rather than as a plausible-looking
single number.

    METHOD 1 — energy-flow diagnostics at the targets   (SF script "approach A")
        ef<species>_tot_ylow is already a POWER [W] through a Y cell face
        (advection + conduction + the sheath sink).  Target power is the sum
        over the radial index at the target face.

    METHOD 2 — analytic sheath transmission             (SF script "approach C")
        q = gamma_e n Te Cs + gamma_i n Ti Cs,  Cs = sqrt((Z Te + Ti)/m_i),
        integrated over the SI face area  dA = J dx dz / sqrt(g_22).

    METHOD 3 — volumetric energy-source integral        (supervisor's suggestion)
        SP<species> [Pa/s] is the *total* pressure source of that species
        (external heating + reactions + sheath sink + collisional exchange).
        Energy density of an ideal gas is (3/2)P, so

            P = (3/2) * sum_cells  SP  *  dV,     dV = J dx dy dz  [m^3]

        No factor of qe anywhere: the dump attribute `conversion` for SP is
        Pnorm*Omega_ci = qe*Tnorm*Nnorm*Omega_ci, so xHermes already hands back
        Pa/s = J m^-3 s^-1 = W m^-3.  The eV -> J conversion is baked into the
        normalisation; multiplying by qe again would be wrong by 1.6e-19.

    METHOD 4 — the actual injected heating power
        Same integral, but over `P<species>_src` = evolve_pressure's
        `final_source`, i.e. ONLY the external source from the input file
        (`[Pe] source`, `[Pd+] source`, times the time-dependent prefactor).
        This is the ground truth "how many watts am I putting in".

    Plus: an effective sheath transmission coefficient gamma_eff back-computed
    from method 1's power and the particle flow pf<ion>_tot_ylow.  For
    sheath_boundary_simple with gamma_e = 4.5, gamma_i = 2.5 this should land
    near 7.

Plotting reuses the polygon routines from ``Hermes-3_postprocessing.py``
(loaded by path — see ``_load_pp``), so the maps look identical to the rest of
the toolchain.  On top of the raw dump variables it can plot derived power
densities: ``Pnet`` (3/2 sum SP), ``Psrc`` (3/2 sum P_src), ``Pother``
(Pnet - Psrc, i.e. everything that is not external heating) and ``dEdt``
(3/2 sum ddt(P)).

--------------------------------------------------------------------------------
TWO GOTCHAS THIS SCRIPT FIXES RELATIVE TO SF_heat_transport_pp_xhermes.py
--------------------------------------------------------------------------------
1. **Upper-target faces are dropped by keep_yboundaries=False.**
   sheath_boundary_simple puts the sheath power on the *ylow face of the first
   guard cell* at an upper target (`electron_sheath_power_ylow[ip]`,
   sheath_boundary_simple.cxx:470) but on the *first domain cell* at a lower
   target (`[i]`, :407).  xHermes defaults to keep_yboundaries=False, which
   deletes the guard column — so reading `ef*_tot_ylow[:, ny-1]` returns the
   *interior* face and misses the entire target load.  On the LSN test case
   that is 58 kW read as 0 W.  This script therefore opens a second, geometry-
   free dataset with keep_yboundaries=True purely to read the correct faces,
   and prints the naive value alongside so the size of the effect is visible.

2. **The radial index mapping in ``_build_patches`` assumes x-guard cells are
   present** (``bout_i = j + 1``, gridue j=1 -> bout_i=2).  Passing a
   ``[MXG:-MXG]``-stripped array (as the reference script's __main__ does)
   shifts the picture radially by MXG and silently drops the outermost rings.
   Here the full-width radial array is passed, which is what the routine wants.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
    python hermes-heat-pp.py <grid_file> <sim_results_dir> [options]

    # numbers only, no figure
    python hermes-heat-pp.py mastu_sf45_bout.grd.nc runs/sf45 --plot False

    # where is the power being deposited / lost?
    python hermes-heat-pp.py mastu_sf45_bout.grd.nc runs/sf45 --var Pother --seps
"""

import argparse
import glob
import importlib.util
import os
import sys
import warnings

import numpy as np
import netCDF4
import xarray as xr
import xhermes
import matplotlib.pyplot as plt


# ── Physical constants (match Hermes-3 / BOUT++ SI:: values) ──────────────────
QE = 1.602176634e-19   # elementary charge [C]
MP = 1.672621898e-27   # proton mass [kg]
ME = 9.1093837015e-31  # electron mass [kg]

# Canonical BOUT++ double-null target slots. A single null fills only the first
# two (its two legs at the poloidal domain ends).
TARGET_SHORT = ["SP1 (NW)", "SP2 (NE)", "SP3 (SE)", "SP4 (SW)"]

# Candidate locations for the shared plotting module, searched in order.
_PP_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "Hermes-3_postprocessing.py"),
    os.path.expanduser("~/Dev/My_working_scripts/BOUT++_Post_Processing/"
                       "Hermes-3_postprocessing.py"),
]

BANNER = "hermes-heat-pp"


def _load_pp(explicit=None):
    """
    Import ``Hermes-3_postprocessing.py`` by path.

    The filename is not a legal Python identifier (hyphen, leading digit in the
    second component), so a normal ``import`` cannot reach it; importlib by
    file location can. Returns None if it cannot be found, in which case the
    script still prints every power number and only the figure is skipped.
    """
    for path in ([explicit] if explicit else []) + _PP_CANDIDATES:
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("hermes3_pp", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"[{BANNER}] Plot routines from {path}")
            return mod
    print(f"[{BANNER}] Could not find Hermes-3_postprocessing.py "
          f"(looked in {_PP_CANDIDATES}); plotting disabled. "
          f"Pass --pp_path to point at it.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def _common_nt(sim_path):
    """
    Largest number of output steps present in *every* BOUT.dmp file.

    Per-rank dumps (especially with the PETSc/SNES solvers) need not flush the
    same number of steps; reading a step a short rank never wrote crashes the
    combine. Returns ``(min_t, max_t, nfiles)``.
    """
    files = sorted(glob.glob(os.path.join(sim_path, "BOUT.dmp.*.nc")))
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files found in {sim_path!r}")
    tlens = []
    for f in files:
        ds = netCDF4.Dataset(f)
        tlens.append(ds.dimensions["t"].size if "t" in ds.dimensions else 0)
        ds.close()
    return min(tlens), max(tlens), len(files)


def _open_clean_grid(grid_file):
    """
    Open a snowflake grid so xbout's toroidal geometry can ingest it.

    These grids carry the BOUT fields (psixy, Rxy, Bpxy, J, g_22, dx, ...) on
    dims ``(x2, y2)`` alongside a corner mesh (rm/zm) on ``(x, y, z)``. xbout
    only recognises ``x/y/z`` and drops everything else — including psixy —
    so a bare ``geometry="toroidal"`` fails. Rename the field dims to ``x/y``
    and hand that over; the raw grid (with rm/zm) is opened separately for
    plotting.
    """
    g = xr.open_dataset(grid_file)
    drop_dims = [d for d in ("x", "y", "z", "t") if d in g.dims]
    field_grid = g.drop_dims(drop_dims).rename(
        {k: v for k, v in {"x2": "x", "y2": "y"}.items() if k in g.dims})
    return field_grid


def _tavg(da, t0, t1):
    """
    Time-average a DataArray over steps [t0, t1] inclusive, collapsing the time
    and toroidal dims so the result is 2D (radial, poloidal).

    This is a mean (mean power over the window), not a dt-weighted integral.
    Variables without a time axis (``fixed_density`` species, metric terms) are
    passed through unchanged.
    """
    sl = da.isel(t=slice(t0, t1 + 1)) if "t" in da.dims else da
    keep = {"x", "theta", "y"}
    reduce_dims = [d for d in sl.dims if d not in keep]   # t + zeta/z
    return sl.mean(reduce_dims).values if reduce_dims else sl.values


def _species_list(ds):
    """
    Species with an evolved pressure, in dump order.

    Detected from the ``SP<name>`` / ``P<name>_src`` diagnostics that
    evolve_pressure writes when ``diagnose = true``.
    """
    names = []
    for v in ds.data_vars:
        if v.startswith("SP") and len(v) > 2:
            names.append(v[2:])
        elif v.startswith("P") and v.endswith("_src"):
            n = v[1:-4]
            if n and n not in names:
                names.append(n)
    # Preserve a sane order: electrons first, then the rest alphabetically.
    return sorted(set(names), key=lambda n: (n != "e", n))


def _charged_species(ds, species):
    """
    Positively charged species present in the run, evolved-pressure ones first.

    Scans the input-file sections rather than only the evolved-pressure list,
    because an ion can be quasineutral / fixed-density (``h+`` with
    ``type = quasineutral, set_temperature``) and still carry the whole target
    load — method 2 needs its AA and charge either way.
    """
    opts = ds.attrs.get("options") or {}
    ions = []
    for name in list(species) + sorted(opts.keys() if hasattr(opts, "keys") else []):
        if name == "e" or name in ions:
            continue
        try:
            charge = float(opts[name]["charge"])
        except Exception:
            continue
        if charge > 0 and (f"N{name}" in ds.data_vars or name in species):
            ions.append(name)
    return ions


def _opt(ds, section, key, default):
    """Read one option out of BOUT.settings, falling back to ``default``."""
    try:
        return float(ds.attrs["options"][section][key])
    except Exception:
        return default


def load_run(sim_path, grid_file, twindow=20):
    """
    Load one Hermes-3 run and return everything the four methods need, in SI.

    Two datasets are opened:

    ``ds``   geometry="toroidal" from the grid, keep_yboundaries=False (the
             xHermes default).  Carries R/Z/psi, the metric, and every field on
             the interior poloidal frame.  This is what the SF script sees.
    ``dsg``  the same dumps with keep_yboundaries=True and **no** grid.  The
             snowflake grids contain no y-boundary guard cells, so xbout cannot
             attach the toroidal geometry to a y-guarded dataset — but the
             energy/particle flow diagnostics do not need geometry, and the
             upper-target faces only exist here (see the module docstring).
             ``None`` if the second open fails; the script degrades gracefully.
    """
    print(f"[{BANNER}] Loading {sim_path}")
    min_t, max_t, nfiles = _common_nt(sim_path)
    if min_t == 0:
        raise ValueError(f"{sim_path}: a dump file has no time steps.")
    if min_t != max_t:
        print(f"[{BANNER}] Time-step mismatch across {nfiles} dump files "
              f"(t lengths {min_t}..{max_t}); clipping to index {min_t - 1}.")

    field_grid = _open_clean_grid(grid_file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xhermes.open(sim_path, geometry="toroidal",
                          gridfilepath=field_grid, info=False)
    ds = ds.isel(t=slice(0, min_t))

    dsg = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dsg = xhermes.open(sim_path, info=False, keep_yboundaries=True)
        dsg = dsg.isel(t=slice(0, min_t))
    except Exception as exc:                              # pragma: no cover
        print(f"[{BANNER}] Could not open a y-guarded view ({exc}); "
              f"upper-target faces will be read from the interior frame "
              f"(the SF-script behaviour) and will under-report.")

    meta = ds.metadata
    nx, ny = int(meta["nx"]), int(meta["ny"])
    ny_inner = int(meta["ny_inner"])
    MXG, MYG = int(meta["MXG"]), int(meta["MYG"])
    single_null = int(meta["jyseps2_1"]) == int(meta["jyseps1_2"])

    t1 = min_t - 1
    t0 = max(0, min_t - twindow)

    # ── Metric: SI cell volume and Y-face area ───────────────────────────────
    J, dx, dy, dz = (ds[k].values for k in ("J", "dx", "dy", "dz"))
    g_22 = ds["g_22"].values
    dV = J * dx * dy * dz                       # [m^3]  full torus (dz = 2*pi)
    dA_y = J * dx * dz / np.sqrt(g_22)          # [m^2]  poloidal-face area

    # ── Targets in the interior (no-y-guard) poloidal frame ──────────────────
    # side: 'lower' = the target is on this cell's ylow face
    #       'upper' = the target is on its yhigh face = the guard cell's ylow
    if single_null:
        targets = [(0, "lower"), (ny - 1, "upper"), None, None]
        print(f"[{BANNER}] Single-null (jyseps2_1 == jyseps1_2): two legs at "
              f"theta = 0 and {ny - 1}.")
    else:
        # Region 1 spans theta 0..ny_inner-1, region 2 theta ny_inner..ny-1, so
        # SP3 is at ny_inner — the FIRST cell of region 2.  (The SF script uses
        # ny_inner + 1, one cell in from the target.)
        targets = [(0, "lower"), (ny_inner - 1, "upper"),
                   (ny_inner, "lower"), (ny - 1, "upper")]

    species = _species_list(ds)
    ions = _charged_species(ds, species)
    print(f"[{BANNER}] Species with an evolved pressure: "
          f"{', '.join(species) if species else '(none — is diagnose = true?)'}"
          f"    ions: {', '.join(ions) if ions else '(none)'}")

    return dict(
        ds=ds, dsg=dsg, meta=meta,
        nx=nx, ny=ny, ny_inner=ny_inner, MXG=MXG, MYG=MYG,
        single_null=single_null, topology=meta.get("topology"),
        t0=t0, t1=t1, nt=min_t,
        dV=dV, dA_y=dA_y, targets=targets,
        species=species, ions=ions,
        sim_path=sim_path, grid_file=grid_file,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Guarded-frame index mapping (fixes the missing upper-target faces)
# ══════════════════════════════════════════════════════════════════════════════

def _guard_map(run):
    """
    Build the interior-poloidal -> y-guarded-poloidal cell index map.

    With ``keep_yboundaries=True`` xbout inserts MYG guard cells at each end of
    each poloidal *region*: one region for a single null (2*MYG extra cells),
    two for a double null / snowflake (4*MYG extra).  The offset is therefore
    MYG below ``ny_inner`` and 3*MYG above it.

    Returns ``(offset_fn, n_regions)`` or ``(None, None)`` if the y-guarded
    dataset is absent or its shape does not match either expectation.
    """
    dsg = run["dsg"]
    if dsg is None:
        return None, None
    ydim = "y" if "y" in dsg.sizes else "theta"
    ny_g, ny, MYG, ny_inner = dsg.sizes[ydim], run["ny"], run["MYG"], run["ny_inner"]
    extra = ny_g - ny
    if extra == 2 * MYG:
        n_reg = 1
        fn = lambda j: j + MYG                                    # noqa: E731
    elif extra == 4 * MYG:
        n_reg = 2
        fn = lambda j: j + (MYG if j < ny_inner else 3 * MYG)     # noqa: E731
    else:
        print(f"[{BANNER}] Unexpected y-guarded size ({ny_g} vs ny={ny}, "
              f"MYG={MYG}); cannot map target faces. Falling back.")
        return None, None
    return fn, n_reg


def _validate_guard_map(run, fn):
    """
    Sanity-check the index map by comparing a field that exists in both views.

    Cheap insurance against a topology whose guard layout differs from the two
    cases handled above: if the mapped columns of ``Te`` (or the first shared
    4D variable) do not agree, the map is rejected and the caller degrades to
    the interior-frame faces.
    """
    ds, dsg = run["ds"], run["dsg"]
    var = next((v for v in ("Te", "Ne", "Pe")
                if v in ds.data_vars and v in dsg.data_vars), None)
    if var is None:
        return True                       # nothing to check against; trust it
    a = _tavg(ds[var], run["t1"], run["t1"])
    b = _tavg(dsg[var], run["t1"], run["t1"])
    cols = [0, run["ny"] // 3, run["ny"] - 1]
    for j in cols:
        if not np.allclose(a[:, j], b[:, fn(j)], rtol=1e-6, atol=0.0,
                           equal_nan=True):
            print(f"[{BANNER}] Guard-cell index map failed validation on "
                  f"{var} at theta={j}; falling back to interior faces.")
            return False
    return True


def _face_index(run, fn, j, side):
    """Poloidal index of a target face in the y-guarded frame."""
    return fn(j) if side == "lower" else fn(j) + 1


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — energy-flow diagnostics at the target faces  (SF approach A)
# ══════════════════════════════════════════════════════════════════════════════

def method1_energy_flow(run):
    """
    Sum the per-cell energy-flow diagnostics ``ef<species>_tot_ylow`` [W] over
    the radial index at each target face.

    Returns a dict with, per target slot:
        power     : corrected total |electron| + |ion| power  [W]
        naive     : the same read on the interior frame, i.e. exactly what
                    SF_heat_transport_pp_xhermes.py reports  [W]
        per_sp    : {species: power [W]}
        flux      : ion particle flow through the face [s^-1] (for gamma_eff)
        corrected : whether the guarded face could be used
    """
    ds, dsg = run["ds"], run["dsg"]
    MXG, sl = run["MXG"], slice(run["MXG"], -run["MXG"] or None)
    fn, _ = _guard_map(run)
    if fn is not None and not _validate_guard_map(run, fn):
        fn = None

    # Time-averaged flow fields, from the guarded view when it is usable.
    src = dsg if fn is not None else ds
    ef, pf = {}, {}
    for name in run["species"]:
        v = f"ef{name}_tot_ylow"
        if v in src.data_vars:
            ef[name] = _tavg(src[v], run["t0"], run["t1"])
    for name in run["ions"]:
        v = f"pf{name}_tot_ylow"
        if v in src.data_vars:
            pf[name] = _tavg(src[v], run["t0"], run["t1"])
    ef_naive = {name: _tavg(ds[f"ef{name}_tot_ylow"], run["t0"], run["t1"])
                for name in run["species"] if f"ef{name}_tot_ylow" in ds.data_vars}

    # Species with NO ef*_tot_ylow face diagnostic contribute zero here, and
    # that is not a gap in the dump. neutral_mixed writes its flows under
    # different names (efd_adv_par_ylow, efd_cond_par_ylow, efd_*_perp_*low)
    # and never an ef<name>_tot_ylow — AND those flows are identically zero on
    # every domain boundary, because neutrals do not leave through the mesh
    # faces at all. neutral_boundary takes their energy volumetrically instead
    # (E<name>_wall_refl / E<name>_target_refl, neutral_boundary.cxx:269-270).
    # So methods 1 and 2 are STRUCTURALLY blind to the neutral wall load; only
    # method 3 sees it.
    missing = [s for s in run["species"] if s not in ef]
    out = dict(per_target=[], corrected=fn is not None,
               have_flow=bool(ef), species=sorted(ef), missing=missing)
    for slot, tgt in enumerate(run["targets"]):
        if tgt is None:
            out["per_target"].append(None)
            continue
        j, side = tgt
        jf = _face_index(run, fn, j, side) if fn is not None else j
        npol = next(iter(a.shape[1] for a in ef.values()), j + 1)
        if not 0 <= jf < npol:                       # never expected; be loud
            print(f"[{BANNER}] Target face index {jf} out of range "
                  f"(0..{npol - 1}) for theta={j}; using the interior face.")
            jf = j
        per_sp = {n: float(np.sum(np.abs(a[sl, jf]))) for n, a in ef.items()}
        naive = float(sum(np.sum(np.abs(a[sl, j])) for a in ef_naive.values()))
        flux = float(sum(np.sum(np.abs(a[sl, jf])) for a in pf.values()))
        out["per_target"].append(dict(
            slot=slot, theta=j, side=side, face=jf,
            per_sp=per_sp, power=float(sum(per_sp.values())),
            naive=naive, flux=flux,
            # per-x profiles kept for the flux-weighted gamma and for plots
            prof={n: np.abs(a[:, jf]) for n, a in ef.items()},
            pf_prof=(np.abs(sum(a[:, jf] for a in pf.values()))
                     if pf else None),
        ))
    out["total"] = sum(t["power"] for t in out["per_target"] if t)
    out["total_naive"] = sum(t["naive"] for t in out["per_target"] if t)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — analytic sheath transmission  (SF approach C)
# ══════════════════════════════════════════════════════════════════════════════

def method2_analytic_sheath(run, gamma_e=None, gamma_i=None):
    """
    Per-species sheath heat flux at the Bohm speed, integrated over face area.

        Cs        = sqrt((Z_i Te + Ti) / m_i)                        [m/s]
        q         = gamma_e Ne Te Cs + gamma_i Ni Ti Cs   (Te,Ti in J)
        P_target  = sum_x q * dA_y

    ``gamma_e``/``gamma_i`` default to the values actually used by
    ``sheath_boundary_simple`` in this run (read from BOUT.settings), falling
    back to the component defaults of 3.5/3.5.  The 5/3-polytropic and enthalpy
    refinements of the full sheath BC are not included.

    Fields are taken at the last *domain* cell (as the SF script does), not at
    the sheath-edge midpoint, so this is a cell-centre estimate.
    """
    ds, sl = run["ds"], slice(run["MXG"], -run["MXG"] or None)
    if gamma_e is None:
        gamma_e = _opt(ds, "sheath_boundary_simple", "gamma_e", 3.5)
    if gamma_i is None:
        gamma_i = _opt(ds, "sheath_boundary_simple", "gamma_i", 3.5)

    ion = run["ions"][0] if run["ions"] else None
    if ion is None or "Te" not in ds.data_vars:
        return dict(per_target=[None] * 4, total=float("nan"),
                    gamma_e=gamma_e, gamma_i=gamma_i, ion=ion, ok=False)

    AA = _opt(ds, ion, "AA", 2.0)
    Z = _opt(ds, ion, "charge", 1.0)
    mi = AA * MP

    Te = _tavg(ds["Te"], run["t0"], run["t1"])                     # [eV]
    Ne = _tavg(ds["Ne"], run["t0"], run["t1"])                     # [m^-3]
    Ti = _tavg(ds[f"T{ion}"], run["t0"], run["t1"]) if f"T{ion}" in ds else Te
    Ni = _tavg(ds[f"N{ion}"], run["t0"], run["t1"]) if f"N{ion}" in ds else Ne

    out = dict(per_target=[], gamma_e=gamma_e, gamma_i=gamma_i, ion=ion,
               AA=AA, Z=Z, ok=True)
    for tgt in run["targets"]:
        if tgt is None:
            out["per_target"].append(None)
            continue
        j = tgt[0]
        TeJ, TiJ = Te[:, j] * QE, Ti[:, j] * QE
        Cs = np.sqrt(np.clip((Z * TeJ + TiJ) / mi, 0.0, None))     # [m/s]
        q_e = gamma_e * Ne[:, j] * TeJ * Cs                        # [W/m^2]
        q_i = gamma_i * Ni[:, j] * TiJ * Cs
        area = run["dA_y"][:, j]
        out["per_target"].append(dict(
            theta=j, q=q_e + q_i, q_e=q_e, q_i=q_i, area=area,
            power=float(np.sum(((q_e + q_i) * area)[sl])),
            power_e=float(np.sum((q_e * area)[sl])),
            power_i=float(np.sum((q_i * area)[sl])),
            Te=Te[:, j], Ti=Ti[:, j], Ne=Ne[:, j], Ni=Ni[:, j], Cs=Cs,
        ))
    out["total"] = sum(t["power"] for t in out["per_target"] if t)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# METHODS 3 & 4 — volume integrals of the pressure sources
# ══════════════════════════════════════════════════════════════════════════════

def _volume_integral(run, varname):
    """
    (3/2) * sum over interior cells of ``varname`` [Pa/s] * dV [m^3]  ->  [W].

    Also returns the 2D power density (3/2)*var [W/m^3] on the full radial
    frame (x guards included) so it can be handed straight to the polygon
    plotter, which expects that layout.
    """
    ds = run["ds"]
    if varname not in ds.data_vars:
        return None, None
    field = _tavg(ds[varname], run["t0"], run["t1"])        # [Pa/s] = [W/m^3]
    dens = 1.5 * field                                      # [W/m^3]
    sl = slice(run["MXG"], -run["MXG"] or None)
    power = float(np.sum((dens * run["dV"])[sl, :]))        # [W]
    return power, dens


def methods34_volumetric(run):
    """
    METHOD 3 : (3/2) integral of SP<species>      — *all* energy sources.
    METHOD 4 : (3/2) integral of P<species>_src   — external heating only.

    The difference (SP - P_src) = (2/3)*energy_source*(3/2) = energy_source is
    everything the physics components put in or take out: reactions, radiation,
    the sheath heat sink, collisional exchange, numerical viscous heating.
    Splitting it out is the single most useful line in a leak hunt.

    Also integrates ddt(P<species>) to show how far the run is from steady
    state, and P<species> itself for the stored thermal energy.
    """
    rows = []
    for name in run["species"]:
        p_all, dens_all = _volume_integral(run, f"SP{name}")
        p_ext, dens_ext = _volume_integral(run, f"P{name}_src")
        p_ddt, dens_ddt = _volume_integral(run, f"ddt(P{name})")
        energy = None
        if f"P{name}" in run["ds"].data_vars:
            P = _tavg(run["ds"][f"P{name}"], run["t1"], run["t1"])   # [Pa]
            sl = slice(run["MXG"], -run["MXG"] or None)
            energy = float(np.sum((1.5 * P * run["dV"])[sl, :]))     # [J]
        rows.append(dict(
            name=name, total=p_all, external=p_ext, ddt=p_ddt, energy=energy,
            other=(None if (p_all is None or p_ext is None) else p_all - p_ext),
            dens_all=dens_all, dens_ext=dens_ext, dens_ddt=dens_ddt,
        ))

    def _sum(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) if vals else None

    sl = slice(run["MXG"], -run["MXG"] or None)
    return dict(
        rows=rows,
        total=_sum("total"), external=_sum("external"),
        other=_sum("other"), ddt=_sum("ddt"), energy=_sum("energy"),
        volume=float(np.sum(run["dV"][sl, :])),
    )


# Volumetric energy diagnostics carry these unit strings in Hermes-3 (the
# `W / m^-3` variant is a typo in evolve_pressure.cxx / neutral_parallel_diffusion,
# not a different unit).
_WM3_UNITS = {"W / m^3", "W m^-3", "W / m^-3"}


def volumetric_channels(run):
    """
    Integrate every per-channel volumetric energy diagnostic in the dump.

    These are the individual terms making up method 3's 'other' column:
    radiation (``R*``), reaction energy transfer (``E*_iz``, ``E*_rec``,
    ``E*_cx``), wall/target reflection, pumping, ``V.grad(P)``, and so on.
    Auto-discovered by unit string rather than by a hard-coded name list, so a
    run with different reactions still gets a full breakdown.

    Note these are per-channel and several are *transfers between species*
    which cancel in the total — the table is for locating a channel, not for
    summing.
    """
    ds, sl = run["ds"], slice(run["MXG"], -run["MXG"] or None)
    skip = {f"SP{s}" for s in run["species"]}
    skip |= {f"P{s}_src" for s in run["species"]}
    skip |= {f"ddt(P{s})" for s in run["species"]}
    rows = []
    for name, da in ds.data_vars.items():
        if name in skip or da.attrs.get("units") not in _WM3_UNITS:
            continue
        try:
            field = _tavg(da, run["t0"], run["t1"])
            if field.ndim != 2:
                continue
            rows.append((name, float(np.sum((field * run["dV"])[sl, :])),
                         da.attrs.get("long_name", "")))
        except Exception:
            continue
    return sorted(rows, key=lambda r: -abs(r[1]))


def sheath_potential_check(run, m2):
    """
    Compare the sheath potential to the ambipolar (zero net current) value.

    sheath_boundary_simple sets the electron sheath velocity to the Boltzmann
    expression ``v_e = -sqrt(Te/(2*pi*me)) * (1-Ge) * exp(-(phi_sh - phi_wall)/Te)``
    and floors ``phi_sh`` at ``phi_wall``.  A current-free sheath requires
    ``v_e = Cs``, i.e.

        (phi/Te)_ambipolar = ln( sqrt(Te/(2*pi*me)) / Cs )

    If the actual ``phi_sh/Te`` sits well BELOW that, the electron channel is
    not current-limited: it drains ``v_e/Cs`` times the ambipolar heat flux,
    which is exactly what an inflated ``gamma_e`` in the table above means.
    That is a symptom of an under-converged ``relax_potential`` phi, not of the
    energy diagnostics.

    Returns one row per target, or None if phi / the y-guarded view is absent.
    """
    ds, dsg = run["ds"], run["dsg"]
    src = dsg if dsg is not None else ds
    fn, _ = _guard_map(run)
    if fn is None or "phi" not in src.data_vars or "Te" not in src.data_vars:
        return None
    ion = m2.get("ion")
    mi = m2.get("AA", 2.0) * MP
    Z = m2.get("Z", 1.0)

    Te = _tavg(src["Te"], run["t0"], run["t1"])
    Ti = (_tavg(src[f"T{ion}"], run["t0"], run["t1"])
          if ion and f"T{ion}" in src.data_vars else Te)
    Ne = _tavg(src["Ne"], run["t0"], run["t1"])
    phi = _tavg(src["phi"], run["t0"], run["t1"])
    sl = slice(run["MXG"], -run["MXG"] or None)

    rows = []
    for tgt in run["targets"]:
        if tgt is None:
            rows.append(None)
            continue
        j, side = tgt
        c = fn(j)                            # last domain cell, guarded frame
        g = c + 1 if side == "upper" else c - 1        # the adjacent guard cell
        if not 0 <= g < Te.shape[1]:
            rows.append(None)
            continue
        tes = 0.5 * (Te[:, c] + Te[:, g])
        tis = 0.5 * (Ti[:, c] + Ti[:, g])
        nes = 0.5 * (Ne[:, c] + Ne[:, g])
        phis = np.maximum(0.5 * (phi[:, c] + phi[:, g]), 0.0)
        tefloor = np.maximum(tes, 1e-5)
        ve = np.sqrt(tes * QE / (2 * np.pi * ME)) * np.exp(-phis / tefloor)
        Cs = np.sqrt(np.clip((Z * tes + tis) * QE / mi, 1e-30, None))
        # density-weighted averages over the interior radial cells
        w = np.clip(nes[sl], 0.0, None)
        if w.sum() <= 0:
            rows.append(None)
            continue
        avg = lambda a: float(np.average(a[sl], weights=w))    # noqa: E731
        with np.errstate(divide="ignore", invalid="ignore"):
            amb = np.log(np.sqrt(tes * QE / (2 * np.pi * ME)) / Cs)
        rows.append(dict(theta=j, Te=avg(tes), phi=avg(phis),
                         ratio=avg(phis / tefloor), amb=avg(amb),
                         ve_cs=avg(ve / Cs)))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Effective sheath transmission coefficient
# ══════════════════════════════════════════════════════════════════════════════

def effective_gamma(run, m1, m2):
    """
    Back-compute gamma from the measured target power and particle flow.

    Hermes writes the sheath heat flux as ``q_s = gamma_s T_s n v``, so with
    Gamma = |pf<ion>_tot_ylow| [s^-1] the particle flow through the same face,

        gamma_s = sum_x |ef_s| / ( qe * sum_x |pf| * T_s )

    using the *flux-weighted* temperature, which is exact if the face flux is
    the sheath flux.  gamma_tot = gamma_e + gamma_i is the number to compare
    against gamma_e + gamma_i from the input file (4.5 + 2.5 = 7 for the SF
    runs).  A gamma_tot far from that means either the face being read is not
    the sheath face, or the flow carries something other than sheath transport.
    """
    ds, sl = run["ds"], slice(run["MXG"], -run["MXG"] or None)
    ion = m2.get("ion")
    Te = _tavg(ds["Te"], run["t0"], run["t1"]) if "Te" in ds.data_vars else None
    Ti = (_tavg(ds[f"T{ion}"], run["t0"], run["t1"])
          if ion and f"T{ion}" in ds.data_vars else Te)
    if Te is None:
        return None

    rows = []
    for tgt in m1["per_target"]:
        if tgt is None or tgt["pf_prof"] is None:
            rows.append(None)
            continue
        j = tgt["theta"]
        Gam = tgt["pf_prof"]                                   # [s^-1] per x
        gtot = float(np.sum(Gam[sl]))
        if gtot <= 0:
            rows.append(None)
            continue
        entry = {"theta": j, "flux": gtot}
        for sp, T in (("e", Te), (ion, Ti)):
            if sp is None or sp not in tgt["per_sp"]:
                continue
            denom = QE * float(np.sum((Gam * T[:, j])[sl]))
            entry[sp] = tgt["per_sp"][sp] / denom if denom > 0 else float("nan")
        entry["total"] = sum(v for k, v in entry.items()
                             if k not in ("theta", "flux", "total"))
        rows.append(entry)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Printing
# ══════════════════════════════════════════════════════════════════════════════

def _w(x, unit="MW"):
    """Format a power in MW (or W), tolerating None/NaN."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "     —    "
    scale = 1e6 if unit == "MW" else 1.0
    return f"{x / scale: .6f}"


def _rule(char="─", n=86):
    print(char * n)


def print_report(run, m1, m2, m34, gam, chan=None, phi_chk=None):
    """The whole terminal report: four methods, then the balance and gamma."""
    ds = run["ds"]
    print()
    _rule("═")
    print(f"  POWER AUDIT — {os.path.abspath(run['sim_path'])}")
    _rule("═")
    print(f"  grid              : {os.path.basename(run['grid_file'])}")
    print(f"  topology          : {run['topology']}   "
          f"({'single null' if run['single_null'] else 'double null / snowflake'})")
    print(f"  nx x ny           : {run['nx']} x {run['ny']}   "
          f"(MXG={run['MXG']}, MYG={run['MYG']}, ny_inner={run['ny_inner']})")
    print(f"  time window       : steps {run['t0']}..{run['t1']} "
          f"of {run['nt']}   (t = {float(ds['t'].values[run['t1']]):.6g} s)")
    print(f"  interior volume   : {m34['volume']:.6g} m^3   "
          f"(dV = J dx dy dz, dz = 2*pi so this is the full torus)")
    if m34["energy"] is not None:
        print(f"  stored thermal E  : {m34['energy']:.6g} J   "
              f"(3/2 integral of P over the domain)")

    # ── METHOD 1 ──────────────────────────────────────────────────────────────
    print()
    _rule("═")
    print("  METHOD 1 — target power from the energy-flow diagnostics "
          "(ef*_tot_ylow) [MW]")
    _rule("═")
    if not m1["have_flow"]:
        print("  No ef*_tot_ylow diagnostics in the dump "
              "(evolve_pressure needs diagnose = true).")
    else:
        sps = m1["species"]
        head = f"  {'target':<11}{'theta':>7}{'face':>6}"
        head += "".join(f"{s:>13}" for s in sps)
        head += f"{'total':>13}{'SF-script':>13}"
        print(head)
        _rule("-")
        for slot, t in enumerate(m1["per_target"]):
            if t is None:
                print(f"  {TARGET_SHORT[slot]:<11}{'—':>7}{'—':>6}"
                      + "".join(f"{'—':>13}" for _ in sps)
                      + f"{'—':>13}{'—':>13}")
                continue
            row = f"  {TARGET_SHORT[slot]:<11}{t['theta']:>7}{t['face']:>6}"
            row += "".join(f"{_w(t['per_sp'].get(s)):>13}" for s in sps)
            row += f"{_w(t['power']):>13}{_w(t['naive']):>13}"
            print(row)
        _rule("-")
        print(f"  {'TOTAL':<11}{'':>7}{'':>6}"
              + "".join(f"{'':>13}" for _ in sps)
              + f"{_w(m1['total']):>13}{_w(m1['total_naive']):>13}")
        if m1.get("missing"):
            print()
            print(f"  (!) NO target-face diagnostic for: "
                  f"{', '.join(m1['missing'])} — contributing ZERO above.")
            print("      neutral_mixed writes efd_adv_par_ylow / "
                  "efd_cond_par_ylow / efd_*_perp_*low,")
            print("      never efd_tot_ylow, and those are identically zero on "
                  "the domain boundary:")
            print("      neutrals never leave through the mesh faces. "
                  "neutral_boundary removes their")
            print("      energy VOLUMETRICALLY instead (E*_wall_refl, "
                  "E*_target_refl). Methods 1 and 2")
            print("      therefore cannot see the neutral wall load at all — "
                  "only method 3 does.")
        if m1["corrected"]:
            miss = m1["total"] - m1["total_naive"]
            print()
            print("  'face'     = poloidal index in the y-guarded frame that was "
                  "actually read.")
            print("  'SF-script' = the same quantity read the way "
                  "SF_heat_transport_pp_xhermes.py reads it")
            print("               (interior frame, keep_yboundaries=False).")
            if abs(miss) > 1e-3 * max(abs(m1["total"]), 1.0):
                print(f"  >> The SF-script route misses {_w(miss)} MW "
                      f"({100 * miss / m1['total']:.1f}% of the target load): "
                      f"sheath power at an")
                print("     UPPER target sits on the ylow face of the first "
                      "guard cell, which")
                print("     keep_yboundaries=False deletes. This is a "
                      "post-processing artefact,")
                print("     NOT a leak in the simulation.")
        else:
            print()
            print("  (!) Could not build a y-guarded view — these ARE the "
                  "SF-script numbers, and")
            print("      upper targets are under-reported. See the module "
                  "docstring.")

    # ── METHOD 2 ──────────────────────────────────────────────────────────────
    print()
    _rule("═")
    print(f"  METHOD 2 — analytic sheath transmission "
          f"(gamma_e = {m2['gamma_e']:g}, gamma_i = {m2['gamma_i']:g}) [MW]")
    _rule("═")
    if not m2["ok"]:
        print("  Needs Te/Ne and at least one ion species; not available.")
    else:
        print(f"  ion = {m2['ion']}  (AA = {m2['AA']:g}, Z = {m2['Z']:g}),  "
              f"Cs = sqrt((Z Te + Ti)/m_i), fields at the last domain cell")
        print("  Assumes BOTH channels leave at the Bohm speed. "
              "sheath_boundary_simple's electron")
        print("  channel is instead a Boltzmann-limited thermal flux "
              "(~exp(-phi/Te)), so method 2")
        print("  reads high wherever the sheath potential is not the "
              "ambipolar one — a ratio far")
        print("  from 1 here is a statement about the sheath model, "
              "not necessarily a leak.")
        print(f"  {'target':<11}{'theta':>7}{'electrons':>15}{'ions':>15}"
              f"{'total':>15}{'vs method 1':>15}")
        _rule("-")
        for slot, t in enumerate(m2["per_target"]):
            if t is None:
                print(f"  {TARGET_SHORT[slot]:<11}{'—':>7}{'—':>15}{'—':>15}"
                      f"{'—':>15}{'—':>15}")
                continue
            m1t = m1["per_target"][slot]
            ratio = ("—" if not m1t or m1t["power"] == 0
                     else f"{t['power'] / m1t['power']:.3f} x")
            print(f"  {TARGET_SHORT[slot]:<11}{t['theta']:>7}"
                  f"{_w(t['power_e']):>15}{_w(t['power_i']):>15}"
                  f"{_w(t['power']):>15}{ratio:>15}")
        _rule("-")
        ratio = ("—" if not m1["total"]
                 else f"{m2['total'] / m1['total']:.3f} x")
        print(f"  {'TOTAL':<11}{'':>7}{'':>15}{'':>15}"
              f"{_w(m2['total']):>15}{ratio:>15}")

    # ── METHOD 3 ──────────────────────────────────────────────────────────────
    print()
    _rule("═")
    print("  METHOD 3 — volumetric energy-source integral  "
          "(3/2) * sum SP * dV  [MW]")
    _rule("═")
    print("  SP<sp> [Pa/s] is evolve_pressure's TOTAL pressure source: the "
          "external source")
    print("  plus (2/3)*energy_source from every other component (reactions, "
          "sheath sink,")
    print("  collisional exchange, ...).  Pa/s = W/m^3 already — xHermes has "
          "applied the")
    print("  `conversion` attribute qe*Tnorm*Nnorm*Omega_ci, so there is no "
          "further eV->J")
    print("  factor to apply.  Splitting off the external part isolates the "
          "sinks:")
    print()
    print(f"  {'species':<10}{'external (P_src)':>20}{'other (SP - P_src)':>22}"
          f"{'total SP':>16}{'d/dt (3/2 P)':>18}")
    _rule("-")
    for r in m34["rows"]:
        print(f"  {r['name']:<10}{_w(r['external']):>20}{_w(r['other']):>22}"
              f"{_w(r['total']):>16}{_w(r['ddt']):>18}")
    _rule("-")
    print(f"  {'TOTAL':<10}{_w(m34['external']):>20}{_w(m34['other']):>22}"
          f"{_w(m34['total']):>16}{_w(m34['ddt']):>18}")

    # ── METHOD 4 ──────────────────────────────────────────────────────────────
    print()
    _rule("═")
    print("  METHOD 4 — ACTUAL INJECTED POWER  "
          "(3/2) * sum P<sp>_src * dV  [MW]")
    _rule("═")
    print("  P<sp>_src is evolve_pressure's `final_source`: the external source "
          "from the")
    print("  input file only (after the time-dependent prefactor, and after "
          "source_only_in_core")
    print("  masking).  This is the ground truth for 'how many watts am I "
          "putting in'.")
    print()
    for r in m34["rows"]:
        if r["external"] is None:
            continue
        print(f"    {r['name']:<8} {r['external'] / 1e6: .6f} MW")
    _rule("-")
    print(f"    {'TOTAL':<8} {(m34['external'] or 0) / 1e6: .6f} MW"
          f"      <<< injected heating power")

    # ── Balance ───────────────────────────────────────────────────────────────
    print()
    _rule("═")
    print("  POWER BALANCE")
    _rule("═")
    P_in = m34["external"]
    P_out = m1["total"] if m1["have_flow"] else None
    ddt = m34["ddt"]
    print(f"    injected           (method 4) : {_w(P_in)} MW")
    print(f"    target load        (method 1) : {_w(P_out)} MW")
    print(f"    target load        (method 2) : {_w(m2.get('total'))} MW")
    print(f"    net volumetric     (method 3) : {_w(m34['total'])} MW")
    print(f"    dE/dt of stored P             : {_w(ddt)} MW"
          f"    (0 = steady state)")

    # -- Steady state first: nothing else is interpretable without it. --------
    if ddt is not None and m34["energy"]:
        tau = abs(m34["energy"] / ddt) if ddt else float("inf")
        frac = abs(ddt) / P_in if P_in else float("inf")
        print()
        if frac > 0.1:
            vs = (f"{100 * frac:.0f}% of the injected power" if np.isfinite(frac)
                  else "nonzero with no external heating at all")
            print(f"  (!) NOT IN STEADY STATE. |dE/dt| is {vs}; the stored")
            print(f"      thermal energy ({m34['energy']:.4g} J) "
                  f"{'drains' if ddt < 0 else 'grows'} on a {tau:.3g} s "
                  f"timescale. Every balance below")
            print("      carries that as a genuine, physical imbalance — do "
                  "not read it as a leak.")
        else:
            print(f"      |dE/dt| is {100 * frac:.1f}% of injected — close "
                  f"enough to steady state to")
            print("      interpret the balance below.")

    # -- Methods 1 and 3 are NOT independent. Spell out how they compose. -----
    if P_in is not None and P_out is not None and m34["total"] is not None:
        other = m34["total"] - P_in + P_out
        print()
        _rule("-")
        print("  Reconciling method 1 against method 3 — they are NOT "
              "independent estimates:")
        _rule("-")
        print("  sheath_boundary_simple applies the target heat sink as a "
              "VOLUMETRIC source in")
        print("  the last cell (electron_energy_source[i] += heatflow/dv) and "
              "writes the SAME")
        print("  `heatflow` to *_sheath_power_ylow. So method 1's target load "
              "is already one of")
        print("  the terms inside method 3's SP integral. They should differ, "
              "by construction:")
        print()
        print(f"      (3/2) integral SP        = {_w(m34['total'])} MW   "
              f"(method 3)")
        print(f"        = injected               {_w(P_in)} MW   (method 4)")
        print(f"          - target load          {_w(P_out)} MW   (method 1, "
              f"a sink)")
        print(f"          + everything else      {_w(other)} MW   <-- all "
              f"other volumetric terms")
        if other > 0.05 * abs(P_in or 1.0):
            print()
            print(f"  (!) 'everything else' is a net GAIN of {_w(other)} MW "
                  f"({100 * other / P_in:.0f}% of injected).")
            print("      Volumetric terms should mostly be SINKS (radiation, "
                  "ionisation cost, CX to")
            print("      cold neutrals). A large positive value means energy "
                  "is being created — see")
            print("      the channel breakdown below to find which term.")

        # -- Net outflow through every boundary, from the exact identity. -----
        if ddt is not None:
            outflow = m34["total"] - ddt
            unacc = outflow - P_out
            print()
            _rule("-")
            print("  Net energy outflow through ALL domain boundaries "
                  "(exact identity):")
            _rule("-")
            print("      (3/2) integral ddt(P) = (3/2) integral SP "
                  "- (net boundary outflow)")
            print(f"      => net outflow        = {_w(outflow)} MW")
            print(f"         of which targets   = {_w(P_out)} MW   (method 1)")
            print(f"         UNACCOUNTED        = {_w(unacc)} MW"
                  + (f"  ({100 * unacc / outflow:+.0f}%)" if outflow else ""))
            print()
            print("      The unaccounted part is radial (x) boundary flow plus "
                  "any term added")
            print("      straight to ddt(P) rather than to Sp: damp_p_nt, "
                  "low_T/low_p_diffuse_perp,")
            print("      hyper_z, numerical viscous heating. NOTE there is no "
                  "ef*_tot_xlow in most")
            print("      dumps (only anomalous_diffusion and recycling "
                  "register x-flows), so radial")
            print("      losses are usually INVISIBLE to method 1 — this is "
                  "the biggest blind spot")
            print("      in the whole audit.")
    elif P_out is not None and not P_in:
        print()
        print("      No external heating in this run (P_src = 0), so the whole "
              "target load is")
        print("      stored energy draining out — compare it against dE/dt "
              "above.")

    # ── Volumetric channel breakdown ─────────────────────────────────────────
    if chan:
        print()
        _rule("═")
        print("  VOLUMETRIC ENERGY CHANNELS  (integral of the W/m^3 "
              "diagnostics over the domain) [MW]")
        _rule("═")
        print("  These make up the 'everything else' term above. Several are "
              "TRANSFERS between")
        print("  species and cancel in the total, so read them individually, "
              "do not sum them.")
        print()
        print(f"  {'diagnostic':<22}{'MW':>13}   description")
        _rule("-")
        for name, val, long_name in chan:
            print(f"  {name:<22}{val / 1e6:>13.6f}   {long_name[:44]}")

        # Wall/target load carried by species that method 1 cannot see.
        wall = sum(v for n, v, _ in chan
                   if n.endswith(("_wall_refl", "_target_refl")))
        if wall:
            _rule("-")
            print(f"  {'wall + target refl':<22}{wall / 1e6:>13.6f}   "
                  f"<-- NEUTRAL wall load, invisible to methods 1 and 2")

        # Does the table account for method 3's 'everything else'? Usually not,
        # because components with diagnose = false write no channel at all.
        if (m34["total"] is not None and m34["external"] is not None
                and m1.get("have_flow")):
            other = m34["total"] - m34["external"] + m1["total"]
            listed = sum(v for _, v, _ in chan)
            _rule("-")
            print(f"  {'sum of listed':<22}{listed / 1e6:>13.6f}   "
                  f"(transfers double-count; indicative only)")
            print(f"  {'everything else':<22}{other / 1e6:>13.6f}   "
                  f"(from the balance above)")
            if abs(other - listed) > 0.05 * abs(m34["external"] or 1.0):
                print()
                print(f"  (!) The table does not account for "
                      f"{_w(other - listed)} MW. Components with")
                print("      diagnose = false write no channel diagnostic at "
                      "all. To close this, enable")
                print("      diagnose for sheath_boundary_simple (E*_sheath), "
                      "braginskii_heat_exchange")
                print("      (E*_coll_*) and the reaction components, then "
                      "re-run.")

    # ── gamma ─────────────────────────────────────────────────────────────────
    print()
    _rule("═")
    print("  EFFECTIVE SHEATH TRANSMISSION COEFFICIENT  "
          "gamma = q / (Gamma * T)")
    _rule("═")
    if not gam or all(g is None for g in gam):
        print("  Needs pf<ion>_tot_ylow (evolve_density diagnose = true) and "
              "Te; not available.")
    else:
        ion = m2.get("ion")
        print(f"  input file: gamma_e = {m2['gamma_e']:g}, "
              f"gamma_i = {m2['gamma_i']:g}  ->  expect gamma_total ~ "
              f"{m2['gamma_e'] + m2['gamma_i']:g}")
        print(f"  {'target':<11}{'Gamma [s^-1]':>16}{'gamma_e':>12}"
              f"{'gamma_' + (ion or 'i'):>12}{'gamma_total':>14}")
        _rule("-")
        for slot, g in enumerate(gam):
            if g is None:
                print(f"  {TARGET_SHORT[slot]:<11}{'—':>16}{'—':>12}"
                      f"{'—':>12}{'—':>14}")
                continue
            ge = g.get("e", float("nan"))
            gi = g.get(ion, float("nan"))
            print(f"  {TARGET_SHORT[slot]:<11}{g['flux']:>16.4e}{ge:>12.3f}"
                  f"{gi:>12.3f}{g['total']:>14.3f}")

    # ── Is the sheath ambipolar? Explains an inflated gamma_e. ───────────────
    if phi_chk and any(r for r in phi_chk):
        print()
        _rule("-")
        print("  Sheath potential vs the ambipolar (zero net current) value:")
        _rule("-")
        print(f"  {'target':<11}{'Te_sh [eV]':>13}{'phi_sh [V]':>13}"
              f"{'phi/Te':>10}{'ambipolar':>12}{'v_e/Cs':>10}")
        for slot, r in enumerate(phi_chk):
            if r is None:
                continue
            print(f"  {TARGET_SHORT[slot]:<11}{r['Te']:>13.3f}{r['phi']:>13.3f}"
                  f"{r['ratio']:>10.2f}{r['amb']:>12.2f}{r['ve_cs']:>10.2f}")
        worst = max((r["ve_cs"] for r in phi_chk if r), default=1.0)
        if worst > 1.5:
            print()
            print(f"  (!) phi_sheath sits BELOW the ambipolar value, so "
                  f"v_e/Cs reaches {worst:.1f}.")
            print("      sheath_boundary_simple sets v_e = "
                  "-sqrt(Te/2*pi*me)*exp(-(phi_sh-phi_wall)/Te) and")
            print("      floors phi_sh at phi_wall, so a too-low phi lets the "
                  "electrons leave at many")
            print("      times the Bohm speed: the sheath carries a large net "
                  "ELECTRON CURRENT and")
            print("      drains v_e/Cs times the ambipolar heat flux. That is "
                  "the inflated gamma_e")
            print("      above. It is a symptom of an under-converged "
                  "relax_potential phi (or of a")
            print("      wall_potential that does not match), NOT of the energy "
                  "diagnostics.")
    print()
    _rule("═")


# ══════════════════════════════════════════════════════════════════════════════
# Derived fields for plotting
# ══════════════════════════════════════════════════════════════════════════════

DERIVED = {
    "Pnet":   ("SP{}",       "net energy source  3/2 SP  [W m$^{-3}$]"),
    "Psrc":   ("P{}_src",    "external heating  3/2 P_src  [W m$^{-3}$]"),
    "dEdt":   ("ddt(P{})",   "d/dt of energy density  3/2 ddt(P)  [W m$^{-3}$]"),
}


def derived_field(run, var, time=None):
    """
    Build a 2D (or 3D, when animating) field for the polygon plotter.

    ``var`` is either a raw dump variable, or one of the derived power
    densities: ``Pnet``, ``Psrc``, ``dEdt`` and ``Pother`` (= Pnet - Psrc),
    optionally restricted to one species with a colon, e.g. ``Pnet:e`` or
    ``Pother:d+``.

    Returns ``(data, label, signed)`` where ``data`` keeps the FULL radial
    extent (x guard cells included) because ``_build_patches`` maps gridue
    j=1 -> bout_i=2, i.e. it expects the guarded layout.  ``signed`` flags a
    field that straddles zero, so the caller can pick a diverging colour map
    with symmetric limits.
    """
    ds = run["ds"]
    base, _, sp = var.partition(":")
    species = [sp] if sp else run["species"]

    def stack(pattern):
        """Sum (3/2)*<pattern> over the requested species; None if absent."""
        acc, found = None, False
        for name in species:
            v = pattern.format(name)
            if v not in ds.data_vars:
                continue
            found = True
            a = ds[v]
            a = a.isel(t=time) if (time is not None and "t" in a.dims) else a
            a = a.isel(zeta=0) if "zeta" in a.dims else a
            arr = 1.5 * a.values
            acc = arr if acc is None else acc + arr
        return acc if found else None

    if base in DERIVED:
        pattern, lab = DERIVED[base]
        data = stack(pattern)
        if data is None:
            raise KeyError(f"No {pattern.format('<species>')} in the dump "
                           f"for species {species}")
        return data, f"{lab}  [{'+'.join(species)}]", True

    if base == "Pother":
        a, b = stack("SP{}"), stack("P{}_src")
        if a is None:
            raise KeyError("No SP<species> in the dump")
        data = a if b is None else a - b
        return (data,
                f"non-external energy source  3/2 (SP - P_src)  "
                f"[W m$^{{-3}}$]  [{'+'.join(species)}]", True)

    if var not in ds.data_vars:
        raise KeyError(f"'{var}' is not a dump variable and not one of "
                       f"{sorted(list(DERIVED) + ['Pother'])}")
    a = ds[var]
    a = a.isel(t=time) if (time is not None and "t" in a.dims) else a
    a = a.isel(zeta=0) if "zeta" in a.dims else a
    units = a.attrs.get("units", "")
    data = a.values
    return data, f"{var}" + (f"  [{units}]" if units else ""), bool(
        np.nanmin(data) < 0 < np.nanmax(data))


def _clim(data, clip=99.5):
    """
    Colour limits, with the outliers clipped back to a percentile.

    Without this the sheath sink cells — O(1e8) W/m^3 concentrated in a couple
    of cells at each target — set the scale and flatten the entire core to a
    single colour, which is exactly the region a leak hunt needs to see.
    ``clip = 100`` restores the true extrema.

    The percentile is taken over the NONZERO magnitudes: source terms are often
    confined to a handful of cells (a sheath sink, a core heating box), and a
    percentile of a mostly-zero field is just zero.  Limits are kept symmetric
    for a field that crosses zero, and one-sided otherwise.
    """
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    lo_raw, hi_raw = float(np.min(finite)), float(np.max(finite))
    nz = np.abs(finite)[np.abs(finite) > 0]
    if clip >= 100 or nz.size == 0:
        return lo_raw, hi_raw
    m = float(np.percentile(nz, clip))
    if m <= 0:
        return lo_raw, hi_raw
    return (max(-m, lo_raw) if lo_raw < 0 else lo_raw,
            min(m, hi_raw) if hi_raw > 0 else hi_raw)


def make_plot(run, pp, args):
    """Static or animated polygon plot of a raw or derived field."""
    grid = xr.open_dataset(run["grid_file"], engine="netcdf4")
    if "rm" not in grid or "zm" not in grid:
        print(f"[{BANNER}] Grid has no rm/zm corner mesh; cannot draw the "
              f"polygon plot.")
        return None
    rm, zm = grid["rm"].values, grid["zm"].values
    ny_inner = int(grid["ny_inner"].values)
    bridge_cut = run["single_null"]

    fig, ax = plt.subplots(figsize=(6, 10))

    if args.animate:
        data, label, signed = derived_field(run, args.var, time=None)
        # (t, x, theta) -> the animator wants (nt, bout_nx, bout_ny)
        anim = pp.animate_plot(fig, ax, rm, zm, data, ny_inner,
                               times=run["ds"]["t"].values,
                               cmap=args.cmap or ("RdBu_r" if signed else "viridis"),
                               colorbar_label=label, interval=args.interval,
                               log=args.log, bridge_cut=bridge_cut)
    else:
        data, label, signed = derived_field(run, args.var, time=args.time)
        # Colour limits from the INTERIOR only. The full-width array is what
        # _build_patches wants (it maps gridue j=1 -> bout_i=2), but the x
        # guard cells it never draws routinely hold junk that would otherwise
        # set the scale.
        interior = data[run["MXG"]:-run["MXG"] or None, :]
        signed = bool(np.nanmin(interior) < 0 < np.nanmax(interior))
        vmin, vmax = ((None, None) if args.log
                      else _clim(interior, args.clip))
        if vmin is not None and args.clip < 100:
            label += f"   (colour clipped at the {args.clip:g}th pct)"
        pp.polygon_plot(ax, rm, zm, data, ny_inner,
                        cmap=args.cmap or ("RdBu_r" if signed else "viridis"),
                        vmin=vmin, vmax=vmax, colorbar_label=label,
                        log=args.log, bridge_cut=bridge_cut)
        ax.set_title(f"{args.var}   (t index {args.time})", fontsize=11)
        anim = None

    if args.branch_cuts:
        pp.plot_branch_cuts(ax, rm, zm, grid, ny_inner)
    if args.seps:
        pp.plot_separatrices(ax, rm, zm, grid, ny_inner)
    plt.tight_layout()

    if args.save:
        if anim is not None:
            print(f"[{BANNER}] Saving animation to {args.save} ...")
            anim.save(args.save)
        else:
            plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[{BANNER}] Saved {args.save}")
    return anim


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _str2bool(value):
    """Parse a CLI boolean so ``--plot False`` works."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Hermes-3 power audit: target load from the energy-flow "
                    "diagnostics (1) and the analytic sheath (2), the "
                    "volumetric SP integral (3), and the actual injected "
                    "power from P_src (4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("grid_file", help="BOUT++ grid file (.nc)")
    parser.add_argument("sim_results", help="Directory of BOUT.dmp.*.nc files")

    parser.add_argument("--twindow", type=int, default=20,
                        help="Number of final timesteps to average over")
    parser.add_argument("--gamma_e", type=float, default=None,
                        help="Electron sheath transmission coefficient for "
                             "method 2 (default: from BOUT.settings, else 3.5)")
    parser.add_argument("--gamma_i", type=float, default=None,
                        help="Ion sheath transmission coefficient for method 2")

    parser.add_argument("--plot", type=_str2bool, default=True,
                        metavar="True/False", help="Draw the polygon map")
    parser.add_argument("--var", default="Pother",
                        help="Field to map: a dump variable, or a derived power "
                             "density Pnet / Psrc / Pother / dEdt, optionally "
                             "restricted to one species as e.g. 'Pother:e'")
    parser.add_argument("--time", type=int, default=-1,
                        help="Time index for the map")
    parser.add_argument("--cmap", default=None,
                        help="Colormap (default: RdBu_r for signed fields, "
                             "viridis otherwise)")
    parser.add_argument("--log", action="store_true",
                        help="Logarithmic colour scale")
    parser.add_argument("--clip", type=float, default=99.5,
                        help="Percentile of |field| used for the symmetric "
                             "colour limits of signed fields; 100 = true "
                             "extrema (the sheath cells then dominate)")
    parser.add_argument("--animate", action="store_true",
                        help="Animate over all common timesteps")
    parser.add_argument("--interval", type=int, default=200,
                        help="Animation frame delay [ms]")
    parser.add_argument("--save", default=None,
                        help="Save the figure/animation here instead of only "
                             "showing it")
    parser.add_argument("--branch_cuts", action="store_true",
                        help="Overlay the topological branch cuts")
    parser.add_argument("--seps", action="store_true",
                        help="Overlay the separatrices")
    parser.add_argument("--pp_path", default=None,
                        help="Explicit path to Hermes-3_postprocessing.py")

    args = parser.parse_args()

    if not os.path.isfile(args.grid_file):
        sys.exit(f"Grid file not found: {args.grid_file}")
    if not os.path.isdir(args.sim_results):
        sys.exit(f"Not a directory: {args.sim_results}")

    run = load_run(args.sim_results, args.grid_file, twindow=args.twindow)

    m1 = method1_energy_flow(run)
    m2 = method2_analytic_sheath(run, gamma_e=args.gamma_e,
                                 gamma_i=args.gamma_i)
    m34 = methods34_volumetric(run)
    gam = effective_gamma(run, m1, m2)
    chan = volumetric_channels(run)
    phi_chk = sheath_potential_check(run, m2)

    print_report(run, m1, m2, m34, gam, chan, phi_chk)

    if args.plot:
        pp = _load_pp(args.pp_path)
        if pp is not None:
            try:
                anim = make_plot(run, pp, args)          # noqa: F841 (keep alive)
            except KeyError as exc:
                print(f"[{BANNER}] {exc}")
            else:
                if not args.save:
                    plt.show()


if __name__ == "__main__":
    main()
