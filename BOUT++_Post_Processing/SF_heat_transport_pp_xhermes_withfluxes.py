#!/usr/bin/env python3
"""
Post-processing for snowflake divertor Hermes-3 simulations — xHermes edition.

This is the SI-clean rewrite of ``SF_heat_transport_pp.py``. The data-loading
layer now goes through **xHermes**, which reads each dump variable's
``conversion`` attribute and returns everything already in SI units. This
removes the whole class of normalisation bugs documented in ``physics.md``:

    - ``efe_tot_ylow``  ->  Watts   (it is a POWER through the cell face,
                                     already integrated over the face area)
    - ``Pe``            ->  Pascals
    - ``Te``            ->  eV,   ``Ne`` -> m^-3
    - ``J, g_22, dx, dz`` -> SI, so J*dx*dz/sqrt(g_22) is m^2 directly

Geometry is built cleanly via xbout's ``geometry="toroidal"`` using the grid's
``psixy``/``Rxy`` etc., so ``R``, ``Z``, ``psi_poloidal``, the topology and the
divertor targets come from xbout (no hand-rolled ``argmax(Rxy)`` midplane). The
snowflake grids store their BOUT fields on dims ``(x2, y2)`` and a separate
corner mesh (``rm``/``zm``) on ``(x, y, z)``; xbout drops variables whose dims it
doesn't recognise, so ``_open_clean_grid`` keeps the field block and renames
``x2->x, y2->y`` before handing the grid to xbout.

Target power is computed by one of two routes (``compute_target_power``):
    default            -> APPROACH A: sum the per-cell energy-flow diagnostics
                          efe_tot_ylow + ef<ion>_tot_ylow (already W); the 1D
                          flux density is power / area.
    --analytic_power   -> APPROACH C: analytic sheath flux q = gamma*n*T*Cs per
                          species, integrated over the SI face area.
Same outputs either way: the 2x2 q_parallel figures (SI and normalised), the
power-fraction table/bar chart, and the --plot_grids mesh render (corner recipe).

Target locations in the BOUT++ double-null poloidal layout (no y-guards frame):
    NW (SP1) : theta = 0
    NE (SP2) : theta = ny_inner - 1
    SE (SP3) : theta = ny_inner
    SW (SP4) : theta = ny - 1
"""

import sys
import os
import glob
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import xarray as xr
import xhermes

# Reuse the INGRID grid-plotting helpers (same routines as visualize_in_grid.py)
# for the --plot_grids panel, instead of duplicating them here.
_VIG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "BOUT++_Post_Processing")
)
if _VIG_DIR not in sys.path:
    sys.path.insert(0, _VIG_DIR)
try:
    import visualize_in_grid as vig
except ImportError:
    vig = None


# Physical constants (match Hermes-3 / BOUT++ SI:: values)
QE = 1.602176634e-19   # elementary charge [C]
MP = 1.672621898e-27   # proton mass [kg]
ME = 9.1093837015e-31  # electron mass [kg]

COLOR_CYCLE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
               "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray"]

TARGET_LABELS = [
    "SP1 – NW target",
    "SP2 – NE target",
    "SP3 – SE target",
    "SP4 – SW target",
]
TARGET_SHORT = ["SP1 (NW)", "SP2 (NE)", "SP3 (SE)", "SP4 (SW)"]


# ── Time handling ──────────────────────────────────────────────────────────────

def _dump_pattern(sim_path):
    squash = os.path.join(sim_path, "BOUT.squash.nc")
    if os.path.exists(squash):
        return squash
    return os.path.join(sim_path, "BOUT.dmp.*.nc")


def _common_nt(sim_path):
    """
    Largest number of output steps present in *every* BOUT.dmp file.

    Per-rank dump files (esp. with the PETSc solver) need not flush the same
    number of steps; reading a step that a short rank never wrote crashes the
    loader. We clip the time window to the last common step. Returns
    ``(min_t, max_t, nfiles)``.
    """
    files = sorted(glob.glob(_dump_pattern(sim_path)))
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files found in {sim_path!r}")
    tlens = []
    for f in files:
        ds = netCDF4.Dataset(f)
        tlens.append(ds.dimensions["t"].size if "t" in ds.dimensions else 0)
        ds.close()
    return min(tlens), max(tlens), len(files)


def _first_unwritten_t(sim_path, min_t,
                       varnames=("efe_tot_ylow", "efd+_tot_ylow", "efi_tot_ylow",
                                 "Pe", "Te", "Ne", "Td+", "Nd+")):
    """
    First output step (< min_t) where any of ``varnames`` still holds the netCDF
    fill value (~9.97e36), i.e. a step that was never fully written. Returns
    ``min_t`` if every step is clean.
    """
    for f in sorted(glob.glob(_dump_pattern(sim_path))):
        with netCDF4.Dataset(f) as d:
            d.set_auto_mask(False)
            for v in varnames:
                if v not in d.variables or "t" not in d[v].dimensions:
                    continue
                a = np.abs(d[v][:min_t]).reshape(min_t, -1)
                bad = np.flatnonzero((a > 9.0e36).any(axis=1))
                if bad.size:
                    min_t = int(bad[0])
    return min_t


def _tavg(da, t0, t1, poloidal_dim="theta"):
    """Time-average a DataArray over steps [t0, t1] (inclusive), collapsing the
    time and toroidal dims so the result is (x, poloidal).

    NOTE: this is an *average* (mean power), not a time integral. See physics.md
    §6 — there is no dt-weighted integration anywhere; the t axis is collapsed.
    """
    sl = da.isel(t=slice(t0, t1 + 1))
    keep = {"x", poloidal_dim}
    reduce_dims = [d for d in sl.dims if d not in keep]   # t + toroidal (zeta/z)
    return sl.mean(reduce_dims).values


# ── Case discovery ─────────────────────────────────────────────────────────────
# Sim folders are named SF<angle>[suffix] or SN[suffix]. The grid for SF<angle> is
# the file in the grid folder whose name contains "sf<angle>" not followed by
# another digit (sf75 -> mastu_sf75p_bout.grd.nc); SN takes the "lsn"/"sn" grid.
# SF-ideal[suffix] (angle "ideal") takes the "sf_pro_max" grid.

_CASE_RE = re.compile(r"^(?:SF(\d+)|(SF-?ideal)|SN)(.*)$", re.IGNORECASE)


def _find_grid(grid_dir, angle):
    if angle is None:
        pat = re.compile(r"(?:^|_)l?sn(?:_|\.)", re.IGNORECASE)
    elif angle == "ideal":
        pat = re.compile(r"sf_pro_max", re.IGNORECASE)
    else:
        pat = re.compile(rf"sf{angle}(?!\d)", re.IGNORECASE)
    hits = sorted(f for f in glob.glob(os.path.join(grid_dir, "*.nc"))
                  if pat.search(os.path.basename(f)))
    if len(hits) > 1:
        raise ValueError(f"several grids match: {[os.path.basename(h) for h in hits]}")
    return hits[0] if hits else None


def _nx(path):
    with netCDF4.Dataset(path) as d:
        return int(d["nx"][...])


def discover_cases(sims_dir, grid_dir, only=None):
    """
    Pair every SF<angle>/SF-ideal/SN sim folder in ``sims_dir`` with its grid in ``grid_dir``.

    Returns a list of dicts (label, path, grid, color, ls) sorted by angle, then
    SF-ideal, SN last.
    Variants of one angle (e.g. SF45-New_Limiter) share its colour, dashed.
    """
    only = {o.upper() for o in only} if only else None
    cases = []
    for name in sorted(os.listdir(sims_dir)):
        path = os.path.join(sims_dir, name)
        m = _CASE_RE.match(name)
        if not os.path.isdir(path) or m is None:
            continue
        if only is not None and name.upper() not in only:
            continue
        dumps = sorted(glob.glob(_dump_pattern(path)))
        if not dumps:
            print(f"Skipping {name}: no BOUT.dmp / BOUT.squash files.")
            continue
        angle = int(m.group(1)) if m.group(1) else "ideal" if m.group(2) else None
        try:
            grid = _find_grid(grid_dir, angle)
        except ValueError as e:
            print(f"Skipping {name}: {e}")
            continue
        if grid is None:
            print(f"Skipping {name}: no matching grid in {grid_dir!r}.")
            continue
        nx_grid, nx_sim = _nx(grid), _nx(dumps[0])
        if nx_grid != nx_sim:
            print(f"Skipping {name}: grid {os.path.basename(grid)} has nx={nx_grid}, "
                  f"sim has nx={nx_sim}.")
            continue
        cases.append({"label": name, "path": path, "grid": grid,
                      "angle": angle, "variant": bool(m.group(3))})
    if only is not None:
        missing = only - {c["label"].upper() for c in cases}
        if missing:
            print(f"--only: not loaded: {sorted(missing)}")
    cases.sort(key=lambda c: (c["angle"] is None, c["angle"] == "ideal",
                              c["angle"] if isinstance(c["angle"], int) else 0,
                              c["label"]))
    bases = list(dict.fromkeys(c["angle"] for c in cases))
    for c in cases:
        c["color"] = COLOR_CYCLE[bases.index(c["angle"]) % len(COLOR_CYCLE)]
        c["ls"] = "--" if c["variant"] else "-"
    print("Cases:")
    for c in cases:
        print(f"  {c['label']:<20s} <- {os.path.basename(c['grid'])}")
    return cases


# ── Grid handling ──────────────────────────────────────────────────────────────

def _open_clean_grid(grid_file):
    """Open a snowflake grid so xbout's toroidal geometry can ingest it.

    These grids carry the BOUT fields (psixy, Rxy, Bpxy, J, g_22, dx, ...) on
    dims ``(x2, y2)`` alongside a corner mesh (rm/zm) on ``(x, y, z)``. xbout
    only recognises ``x/y/z`` and drops everything else — including psixy — which
    is why a bare ``geometry="toroidal"`` fails. We rename the field dims to
    ``x/y`` so xbout sees a normal grid, but the corner mesh is needed for
    real-space plotting, so we extract it *before* dropping its dims rather than
    throwing it away.

    Returns
    -------
    field_grid : xarray.Dataset
        Clean field grid (psixy/Rxy/... on x/y) to hand to xbout.
    corner_mesh : xarray.Dataset
        The variables living on the corner-mesh dims (rm/zm on x/y/z), kept for
        plotting. Empty if the grid has no such variables.
    """
    g = xr.open_dataset(grid_file)
    drop_dims = [d for d in ("x", "y", "z", "t") if d in g.dims]

    # Extract anything on the dims we're about to drop (the corner mesh: rm/zm).
    corner_vars = [v for v in g.variables if set(g[v].dims) & set(drop_dims)]
    corner_mesh = g[corner_vars] if corner_vars else g[[]]

    field_grid = g.drop_dims(drop_dims).rename(
        {k: v for k, v in {"x2": "x", "y2": "y"}.items() if k in g.dims})
    return field_grid, corner_mesh


# ── Loading: returns SI building blocks ─────────────────────────────────────────

def load_simulation(sim_path, grid_file, t_window=20, ben_example=False,
                    ben_strict=False, gamma_e=None, gamma_i=None):
    """
    Load one Hermes-3 simulation via xHermes and return SI building blocks.

    The returned dict contains *only* SI-correct quantities; it does NOT compute
    a target power — that is deferred to ``compute_target_power`` (the A/B/C
    decision point).

    Returns
    -------
    dict with keys:
        d_sep      : (nx,) distance from primary separatrix at LFS midplane [rho_s0]
        q_0        : reference heat flux n0*Te0*cs0 at separatrix midplane [W/m^2]
        target_y   : list of 4 poloidal indices [NW, NE, SE, SW] (no-guard frame);
                     NE, SE are None in a single null (their slices are zeros)
        topology   : xbout topology string (e.g. upper-disconnected-double-null)
        # per-target lists (each a (nx,) array over the radial index):
        efe_power  : electron POWER through each target face   [W]   (efe_tot_ylow)
        ion_power  : ion POWER through each target face         [W]   (efd+_tot_ylow)
        Pe         : electron pressure at the target slice      [Pa]
        Te / Ti    : electron / ion temperature at the target   [eV]
        Ne / Ni    : electron / ion density at the target        [m^-3]
        area       : physical y-face area at the target slice   [m^2]
        corner_mesh : xarray.Dataset of rm/zm corner-mesh vars (for plotting)
        # diagnostics:
        ion_species   : name of the ion energy-flow var used, or None
        ion_name      : ion species name (e.g. "d+")
        ion_AA, ion_Z : ion mass number and charge (for the analytic sheath, C)
        has_sheath_pp : whether sheath-power diagnostics are present (approach B)
        ixseps1, MXG  : topology indices
        balance       : compute_balance dict at the last clean step if
                        ben_example (approach D), else None
    """
    print(f"  Loading {sim_path} ...")

    # Clip to the last common step across ranks, then open in SI (unnormalise=True).
    min_t, max_t, nfiles = _common_nt(sim_path)
    if min_t == 0:
        raise ValueError(f"{sim_path}: a dump file has no time steps; cannot average.")
    if min_t != max_t:
        print(f"    Time-step mismatch across {nfiles} dump files "
              f"(t lengths {min_t}..{max_t}); clipping to t index {min_t - 1}.")
    clean_t = _first_unwritten_t(sim_path, min_t)
    if clean_t != min_t:
        print(f"    Unwritten (netCDF fill) data from t index {clean_t}; "
              f"clipping to t index {clean_t - 1}.")
        min_t = clean_t
    if min_t == 0:
        raise ValueError(f"{sim_path}: no fully written time steps; cannot average.")

    # Clean toroidal geometry from the grid's psixy/Rxy. keep_yboundaries=False
    # (default) so the poloidal frame matches the grid (no y-guards). The corner
    # mesh (rm/zm) is kept aside for real-space plotting.
    field_grid, corner_mesh = _open_clean_grid(grid_file)
    ds = xhermes.open_hermesdataset(datapath=_dump_pattern(sim_path),
                                    inputfilepath=os.path.join(sim_path, "BOUT.settings"),
                                    geometry="toroidal", gridfilepath=field_grid, info=False)
    ds = ds.isel(t=slice(0, min_t))

    meta   = ds.metadata
    ydim     = meta.get("bout_ydim", "theta")   # poloidal dim name ("theta")
    ixseps1  = int(meta["ixseps1"])
    MXG      = int(meta["MXG"])
    ny_inner = int(meta["ny_inner"])
    ny       = int(meta["ny"])
    rho_s0   = float(meta["rho_s0"])

    t0 = max(0, min_t - t_window)
    t1 = min_t - 1

    # ── SI fields, time+toroidal averaged → (nx, npol) ───────────────────────
    efe = _tavg(ds["efe_tot_ylow"], t0, t1, ydim)   # [W] electron power thru face
    Pe  = _tavg(ds["Pe"],  t0, t1, ydim)            # [Pa]
    Te  = _tavg(ds["Te"],  t0, t1, ydim)            # [eV]
    Ne  = _tavg(ds["Ne"],  t0, t1, ydim)            # [m^-3]

    # Ion energy-flow diagnostic (approach A). Prefer the single-ion d+ total.
    ion_species = None
    for cand in ("efd+_tot_ylow", "efi_tot_ylow"):
        if cand in ds:
            ion_species = cand
            break
    ion = _tavg(ds[ion_species], t0, t1, ydim) if ion_species else np.zeros_like(efe)

    # Ion name (e.g. "d+") and its mass/charge — for the analytic sheath (C).
    ion_name = ion_species[2:-len("_tot_ylow")] if ion_species else "d+"
    ion_AA, ion_Z = 2.0, 1.0
    opts = ds.attrs.get("options")
    try:
        ion_AA = float(opts[ion_name]["AA"])
        ion_Z = float(opts[ion_name]["charge"])
    except Exception:
        pass

    # Ion temperature/density at the targets (for C); fall back to electron values.
    Ti = _tavg(ds["T" + ion_name], t0, t1, ydim) if ("T" + ion_name) in ds else Te
    Ni = _tavg(ds["N" + ion_name], t0, t1, ydim) if ("N" + ion_name) in ds else Ne

    has_sheath_pp = any("sheath_power" in v for v in ds.data_vars)

    # ── SI metric → physical y-face area  dA_y = J*dx*dz/sqrt(g_22)  [m^2] ────
    J    = ds["J"].values
    g_22 = ds["g_22"].values
    dx   = ds["dx"].values
    dz   = ds["dz"].values
    area_2d = J * dx * dz / np.sqrt(g_22)     # [m^2]

    # ── R coordinate and LFS midplane (from clean toroidal geometry) ─────────
    R = ds["R"].values                        # (nx, npol) [m]
    y_mid = int(np.argmax(R[ixseps1, :]))     # outboard (max-R) midplane
    R_mid = R[:, y_mid]
    d_sep = (R_mid - R_mid[ixseps1]) / rho_s0

    # ── Reference heat flux q_0 = n0 * Te0 * cs0 (all SI, at sep midplane) ────
    # Use the actual local Ne/Te rather than assuming n = Nnorm.
    n0      = Ne[ixseps1, y_mid]              # [m^-3]
    Te0_J   = Te[ixseps1, y_mid] * QE         # eV -> J
    cs0     = np.sqrt(Te0_J / MP)             # [m/s]
    q_0     = n0 * Te0_J * cs0                # [W/m^2]

    # ── Approach D: balance at the last clean step only (CLAUDE_BEN_FLUXES.md §4) ─
    balance = None
    if ben_example:
        print(f"    Approach D: last clean step, t index {min_t - 1}.")
        balance = compute_balance(ds.isel(t=-1), ydim, meta, opts, ion_name,
                                  ion_AA, ion_Z, gamma_e=gamma_e, gamma_i=gamma_i,
                                  ben_strict=ben_strict)

    # ── Slice the building blocks at each target poloidal index ──────────────
    target_y = [0, ny_inner - 1, ny_inner, ny - 1]
    if int(meta["jyseps2_1"]) == int(meta["jyseps1_2"]):
        target_y = [0, None, None, ny - 1]

    def col(arr2d, yi):
        return np.zeros(arr2d.shape[0]) if yi is None else arr2d[:, yi]

    return dict(
        d_sep=d_sep, q_0=q_0, target_y=target_y,
        topology=meta.get("topology"),
        efe_power=[col(efe, yi) for yi in target_y],
        ion_power=[col(ion, yi) for yi in target_y],
        Pe=[col(Pe, yi) for yi in target_y],
        Te=[col(Te, yi) for yi in target_y],
        Ne=[col(Ne, yi) for yi in target_y],
        Ti=[col(Ti, yi) for yi in target_y],
        Ni=[col(Ni, yi) for yi in target_y],
        area=[col(area_2d, yi) for yi in target_y],
        corner_mesh=corner_mesh,
        ion_species=ion_species, ion_name=ion_name,
        ion_AA=ion_AA, ion_Z=ion_Z,
        has_sheath_pp=has_sheath_pp,
        ixseps1=ixseps1, MXG=MXG,
        balance=balance,
    )


# ── SI sanity report ─────────────────────────────────────────────────────────

def print_si_sanity(datasets, labels):
    """Print SI magnitudes so the units foundation can be eyeballed for sanity."""
    print("\n" + "=" * 78)
    print("SI sanity check — peak magnitudes of the building blocks (interior cells)")
    print("=" * 78)
    hdr = f"{'geometry':<10}{'efe[kW]':>10}{'ion[kW]':>10}{'Pe[Pa]':>10}" \
          f"{'Te[eV]':>10}{'area[cm^2]':>12}{'q_0[MW/m2]':>12}"
    print(hdr)
    print("-" * 78)
    for d, lab in zip(datasets, labels):
        g = d["MXG"]
        sl = slice(g, -g if g else None)
        efe_pk = max(np.nanmax(np.abs(a[sl])) for a in d["efe_power"]) / 1e3
        ion_pk = max(np.nanmax(np.abs(a[sl])) for a in d["ion_power"]) / 1e3
        pe_pk  = max(np.nanmax(np.abs(a[sl])) for a in d["Pe"])
        te_pk  = max(np.nanmax(np.abs(a[sl])) for a in d["Te"])
        ar_pk  = max(np.nanmax(np.abs(a[sl])) for a in d["area"]) * 1e4
        print(f"{lab:<10}{efe_pk:>10.3g}{ion_pk:>10.3g}{pe_pk:>10.3g}"
              f"{te_pk:>10.3g}{ar_pk:>12.3g}{d['q_0']/1e6:>12.3g}")
    print("=" * 78)


# ── Target power: approach A (default) or C (--analytic_power) ──────────────────

def compute_target_power(data, analytic=False, gamma_e=3.5, gamma_i=3.5, ben=False):
    """
    Turn the SI building blocks into a parallel heat-flux profile and an
    integrated power at each target, by one of two routes (see physics.md §7).

    analytic=False  →  APPROACH A (energy-flow diagnostics)
        The per-cell electron/ion energy flows efX_tot_ylow are already POWERS
        [W] through the face (advection + conduction). Target power is just their
        sum over x; the 1D flux density is recovered as q = power / area.
            q[W/m^2]   = (|efe_power| + |ion_power|) / area
            P_target[W] = sum_x (|efe_power| + |ion_power|)

    analytic=True   →  APPROACH C (analytic sheath transmission)
        Per-species sheath heat flux with the Bohm speed, integrated over area.
            Cs       = sqrt((Z_i*Te + Ti) / m_i)                 [m/s]
            q[W/m^2] = gamma_e*Ne*Te + gamma_i*Ni*Ti, each * Cs  (Te,Ti in J)
            P_target[W] = sum_x q * area
        gamma_e, gamma_i are the sheath heat-transmission coefficients (Hermes
        sheath_boundary_simple defaults to 3.5 for both). The 5/3-polytropic and
        enthalpy refinements of the full sheath BC are not included here.

    ben=True        →  APPROACH D (Ben_example_fluxes, CLAUDE_BEN_FLUXES.md)
        Sheath power per cell from compute_balance (last clean step), taken
        from data["balance"]; the analytic/gamma arguments are ignored.
            q[W/m^2]    = (power_e + power_i) / area
            P_target[W] = sum_x (power_e + power_i)

    All routes write the same downstream keys into `data`:
        q        : list of 4 (nx,) arrays, parallel heat flux  [MW/m^2]
        q_norm   : list of 4 (nx,) arrays, q / q_0             [dimensionless]
        P_targets: list of 4 floats, integrated power per target [MW]
        P_total  : total power across the 4 targets             [MW]
        fractions: list of 4 floats, power fraction per target  [%]
        approach : "A (energy-flow)", "C (analytic sheath)" or "D (Ben_example_fluxes)"
    """
    g = data["MXG"]
    sl = slice(g, -g if g else None)          # interior radial cells
    q_0 = data["q_0"]                          # [W/m^2]

    q_list, P_list = [], []
    for i in range(4):
        area = data["area"][i]
        if ben:
            area = data["balance"]["slot_area"][i]
            q = np.divide(data["balance"]["slot_power"][i], area,
                          out=np.zeros_like(area), where=area > 0)       # [W/m^2]
        elif analytic:
            mi  = data["ion_AA"] * MP
            Zi  = data["ion_Z"]
            TeJ = data["Te"][i] * QE
            TiJ = data["Ti"][i] * QE
            Cs  = np.sqrt(np.clip((Zi * TeJ + TiJ) / mi, 0.0, None))   # [m/s]
            q   = gamma_e * data["Ne"][i] * TeJ * Cs \
                + gamma_i * data["Ni"][i] * TiJ * Cs                   # [W/m^2]
        else:
            power_cell = np.abs(data["efe_power"][i]) + np.abs(data["ion_power"][i])
            q = np.divide(power_cell, area,
                          out=np.zeros_like(area), where=area > 0)    # [W/m^2]
        q_list.append(q)
        P_list.append(float(np.sum((q * area)[sl])))                  # [W]

    P_total = sum(P_list)
    data["q"]        = [q / 1e6 for q in q_list]                      # MW/m^2
    data["q_norm"]   = [q / q_0 for q in q_list]                      # dimensionless
    data["P_targets"] = [P / 1e6 for P in P_list]                     # MW
    data["P_total"]  = P_total / 1e6                                  # MW
    data["fractions"] = [100.0 * P / P_total if P_total else 0.0 for P in P_list]
    if ben:
        data["approach"] = ("D (Ben_example_fluxes, ben_strict)"
                            if data["balance"]["strict"] else "D (Ben_example_fluxes)")
    else:
        data["approach"] = "C (analytic sheath)" if analytic else "A (energy-flow)"
    return data


def print_power_table(datasets, labels):
    """Power fraction at each target plus totals (cf. the old script)."""
    col = 13
    width = 12 + col * len(labels)
    print("\n" + "=" * width)
    print(f"Power fraction at each target [%]   —   approach {datasets[0]['approach']}")
    print("=" * width)
    print(f"{'Target':<12}" + "".join(f"{l:>{col}}" for l in labels))
    print("-" * width)
    for ti, short in enumerate(TARGET_SHORT):
        row = f"{short:<12}" + "".join(f"{d['fractions'][ti]:>{col-1}.1f}%" for d in datasets)
        print(row)
    print("=" * width)
    print(f"\n{'Total power [MW]':<16}" + "".join(f"{d['P_total']:>{col}.3e}" for d in datasets))
    print(f"{'q_0 [MW/m^2]':<16}"      + "".join(f"{d['q_0']/1e6:>{col}.3e}"   for d in datasets))


def print_balance_table(datasets, labels):
    """Approach D power and particle balance (CLAUDE_BEN_FLUXES.md), one column per case."""
    col = 13
    width = 28 + col * len(labels)
    bals = [d["balance"] for d in datasets]
    slots = [[0, 1, 2, 3] if len(b["targets"]) == 4 else [0, None, None, 1] for b in bals]

    def row(name, values, fmt=".4g"):
        print(f"{name:<28}" + "".join(f"{v:>{col}{fmt}}" for v in values))

    def per_target(key, ti, scale):
        return [b[key][s[ti]] * scale if s[ti] is not None else np.nan
                for b, s in zip(bals, slots)]

    def parts(key):
        return list(dict.fromkeys(n for b in bals for n in b[key]))

    print("\n" + "=" * width)
    print(f"Power balance [kW], last clean step   —   approach {datasets[0]['approach']}")
    print("=" * width)
    print(f"{'Term':<28}" + "".join(f"{l:>{col}}" for l in labels))
    print("-" * width)
    row("gamma_e", [b["gamma_e"] for b in bals])
    row("gamma_i", [b["gamma_i"] for b in bals])
    row("P_core (sources)", [b["P_core"] / 1e3 for b in bals])
    row("P_x in, core", [b["P_x_in_core"] / 1e3 for b in bals])
    row("P_x in, PF", [b["P_x_in_pf"] / 1e3 for b in bals])
    row("P_x out", [b["P_x_out"] / 1e3 for b in bals])
    for ti, short in enumerate(TARGET_SHORT):
        row(f"Sheath {short} e", per_target("P_sheath_e", ti, 1e-3))
        row(f"Sheath {short} i", per_target("P_sheath_i", ti, 1e-3))
    row("P_rad", [b["P_rad"] / 1e3 for b in bals])
    for n in parts("P_rad_parts"):
        row(f"  {n}", [b["P_rad_parts"].get(n, np.nan) / 1e3 for b in bals])
    row("P_neutral", [b["P_neutral"] / 1e3 for b in bals])
    for n in parts("P_neutral_parts"):
        row(f"  {n}", [b["P_neutral_parts"].get(n, np.nan) / 1e3 for b in bals])
    print("-" * width)
    row("P_net", [b["P_net"] / 1e3 for b in bals])
    row("dE/dt internal", [b["dEdt_int"] / 1e3 for b in bals])
    row("dE/dt kinetic", [b["dEdt_kin"] / 1e3 for b in bals])
    row("Residual P_net - dE/dt", [b["residual"] / 1e3 for b in bals])
    row("Residual [% of P_core]",
        [100.0 * b["residual"] / b["P_core"] if b["P_core"] else np.nan for b in bals], ".3f")
    print("=" * width)

    print(f"\n{'Particle balance [s^-1]':<28}" + "".join(f"{l:>{col}}" for l in labels))
    print("-" * width)
    row("Gamma_x in, core", [b["G_x_in_core"] for b in bals], ".3e")
    row("Gamma_x in, PF", [b["G_x_in_pf"] for b in bals], ".3e")
    row("Gamma_x out", [b["G_x_out"] for b in bals], ".3e")
    for ti, short in enumerate(TARGET_SHORT):
        row(f"Sheath {short}", per_target("G_sheath", ti, 1.0), ".3e")
    row("Ionisation", [b["G_iz"] for b in bals], ".3e")
    row("Recombination", [b["G_rec"] for b in bals], ".3e")
    row("Recycling fraction (Ben)", [b["f_recycle_ben"] for b in bals], ".4f")
    row("Recycling fraction (direct)", [b["f_recycle_direct"] for b in bals], ".4f")
    print("=" * width)


# ── Approach D (Ben_example_fluxes): radial flows, sheath, regions ────────────

def div_a_grad_perp_flows(J, g11, dx, dy, dz, a, f, ben_strict=False):
    """
    Radial flow through every x cell face, Div_a_Grad_perp_upwind from
    analyse_fluxes.py (CLAUDE_BEN_FLUXES.md §3 items 1-2).

    All inputs are (nx, ny) SI arrays including the x guard cells. Returns
    (F_L, F_R): flow into each cell from x-1 and out of it to x+1, already
    integrated over the face ([W] for heat, [s^-1] for particles). Hermes
    upwinds the face coefficient (div_ops.cxx:1496); ben_strict averages it.
    """
    gradient = ((J[:-1] * g11[:-1] + J[1:] * g11[1:])
                * (f[1:] - f[:-1]) / (dx[:-1] + dx[1:]))
    if ben_strict:
        a_face = 0.5 * (a[:-1] + a[1:])
    else:
        a_face = np.where(gradient > 0, a[1:], a[:-1])
    flux = -gradient * a_face * dy[:-1] * dz[:-1]

    F_L = np.zeros_like(f)
    F_R = np.zeros_like(f)
    F_R[:-1] = flux
    F_L[1:] = flux
    return F_L, F_R


def _limit_free(fm, fc, mode):
    """Hermes limitFree (sheath_boundary_simple.cxx:29): guard value from the
    boundary cell fc and its inner neighbour fm. Mode 0: exponential, Neumann
    if that would increase; 1: exponential; 2: linear."""
    with np.errstate(divide="ignore", invalid="ignore"):
        fp = 2.0 * fc - fm if mode == 2 else fc**2 / fm
    fp = np.where(fm > 0, fp, fc)
    if mode == 0:
        fp = np.where(fm < fc, fc, fp)
    return fp


def sheath_target_flows(Ne, Ni, Te, Ti, Vi, area, y, side, gamma_e, gamma_i,
                        AA=2.0, Zi=1.0, sheath_ion_polytropic=1.0,
                        density_mode=1, temperature_mode=1, ben_strict=False):
    """
    Sheath power and particle flow through one target face, for the boundary
    cell row ``y`` on ``side`` ("lower" or "upper").

    Ne, Ni [m^-3], Te, Ti [eV], Vi [m/s] are (nx, ny) SI arrays (no y guards);
    ``area`` is the (nx,) y-face area J*dx*dz/sqrt(g_22) [m^2].
    Returns (power_e, power_i, particles), each (nx,), positive out of the
    domain, in [W], [W], [s^-1].

    Hermes (default, CLAUDE_BEN_FLUXES.md §3 items 3-5): limitFree guards, ion
    speed max(Vi into the target, Cs), q = gamma*T*n*v for both species,
    electron flux = Zi * ion flux (zero current, phi not evolved).
    ben_strict: analyse_fluxes.py exactly — exponential guards, v = Cs,
    q_e = ((gamma_e - 2.5)*T - me*v^2/2)*n*v at lower targets and
    (gamma_e*T - me*v^2/2)*n*v at upper targets.
    """
    yn = y + 1 if side == "lower" else y - 1
    ne_c, ni_c, te_c, ti_c = Ne[:, y], Ni[:, y], Te[:, y], Ti[:, y]

    if ben_strict:
        ne_g = ne_c**2 / Ne[:, yn]
        te_g = te_c**2 / Te[:, yn]
        ti_g = ti_c**2 / Ti[:, yn]
    else:
        ne_g = _limit_free(Ne[:, yn], ne_c, density_mode)
        ni_g = _limit_free(Ni[:, yn], ni_c, density_mode)
        te_g = _limit_free(Te[:, yn], te_c, temperature_mode)
        ti_g = _limit_free(Ti[:, yn], ti_c, temperature_mode)

    nesheath = 0.5 * (ne_c + ne_g)
    tesheath = 0.5 * (te_c + te_g)
    tisheath = 0.5 * (ti_c + ti_g)
    Cs = np.sqrt((sheath_ion_polytropic * QE * tisheath + Zi * QE * tesheath)
                 / (AA * MP))

    if ben_strict:
        v = Cs
        e_coef = gamma_e - 2.5 if side == "lower" else gamma_e
        q_e = (e_coef * QE * tesheath - 0.5 * ME * v**2) * nesheath * v
        q_i = gamma_i * QE * tisheath * nesheath * v
        particles = nesheath * v * area
    else:
        nisheath = 0.5 * (ni_c + ni_g)
        v_in = -Vi[:, y] if side == "lower" else Vi[:, y]
        ion_flux = nisheath * np.maximum(v_in, Cs)
        q_e = gamma_e * QE * tesheath * Zi * ion_flux
        q_i = gamma_i * QE * tisheath * ion_flux
        particles = ion_flux * area

    return q_e * area, q_i * area, particles


def sheath_targets(ny_inner, ny, single_null=False):
    """(y, side) of each target face, no-y-guard frame (CLAUDE_BEN_FLUXES.md §2.1)."""
    if single_null:
        return [(0, "lower"), (ny - 1, "upper")]
    return [(0, "lower"), (ny_inner - 1, "upper"), (ny_inner, "lower"), (ny - 1, "upper")]


def core_y_mask(snowflake_type, ny, jyseps1_1, jyseps2_1, jyseps1_2, jyseps2_2):
    """Poloidal cells whose inner x-boundary is core (CLAUDE_BEN_FLUXES.md §2.2);
    the rest of the inner x-boundary is private flux."""
    y = np.arange(ny)
    inner = (y > jyseps1_1) & (y <= jyseps2_1)
    middle = (y > jyseps2_1) & (y <= jyseps1_2)
    outer = (y > jyseps1_2) & (y <= jyseps2_2)
    if snowflake_type == "sf_plus_high_field_side":
        return middle
    if snowflake_type in ("sf_minus_low_field_side", "sf_minus_high_field_side"):
        return inner | outer
    return inner


def _opt(opts, section, key, default):
    try:
        return float(opts[section][key])
    except Exception:
        return default


def compute_balance(ds_t, ydim, meta, opts, ion_name="d+", ion_AA=2.0, ion_Z=1.0,
                    gamma_e=None, gamma_i=None, ben_strict=False):
    """
    Approach D (Ben_example_fluxes): power and particle balance at one time
    step — analyse_fluxes.py adapted to the SF topology (CLAUDE_BEN_FLUXES.md).

    ``ds_t`` is the xHermes dataset at a single time (SI, x guards kept, no y
    guards). gamma_e / gamma_i default to [sheath_boundary_simple] in
    BOUT.settings (Hermes default 3.5). Powers in [W], particle flows in
    [s^-1]. Inner x-boundary terms are positive into the domain; outer
    x-boundary and sheath terms are positive out of it.

    The per-target sheath arrays are also returned in the 4 target slots
    [NW, NE, SE, SW] (``slot_power``, ``slot_area``) for compute_target_power;
    a single null fills NW and SW and leaves NE, SE at zero.
    """
    def xy(name):
        da = ds_t[name]
        return da.mean([d for d in da.dims if d not in ("x", ydim)]).values

    def has(name):
        return name in ds_t

    g = int(meta["MXG"])
    xs = slice(g, -g if g else None)
    xin, xout = g, ds_t.sizes["x"] - g - 1
    ny, ny_inner = int(meta["ny"]), int(meta["ny_inner"])
    j11, j21, j12, j22 = (int(meta[k]) for k in
                          ("jyseps1_1", "jyseps2_1", "jyseps1_2", "jyseps2_2"))
    single_null = j21 == j12

    J, g11, g_22 = xy("J"), xy("g11"), xy("g_22")
    dx, dy, dz = xy("dx"), xy("dy"), xy("dz")
    dV = J * dx * dy * dz
    area = J * dx * dz / np.sqrt(g_22)

    def vol(f):
        return float(np.sum((f * dV)[xs]))

    neutral = "d"
    try:
        neutral = str(opts[ion_name]["recycle_as"]).strip()
    except Exception:
        pass

    Ne, Te = xy("Ne"), xy("Te")
    Ni = xy("N" + ion_name) if has("N" + ion_name) else Ne
    Ti = xy("T" + ion_name) if has("T" + ion_name) else Te
    Vi = xy("V" + ion_name) if has("V" + ion_name) else np.zeros_like(Ne)

    def coef(kind, sp):
        name = f"anomalous_{kind}_{sp}"
        return xy(name) if has(name) else np.zeros_like(Ne)

    # ── Radial (x) flows: anomalous chi and D, full x (CLAUDE_BEN_FLUXES.md §2.2-2.3)
    d_factor = 1.0 if ben_strict else 1.5
    F_heat_L, F_heat_R = np.zeros_like(Ne), np.zeros_like(Ne)
    for sp, N, T in (("e", Ne, Te), (ion_name, Ni, Ti)):
        for a, f in ((coef("Chi", sp) * N, QE * T),
                     (d_factor * QE * coef("D", sp) * T, N)):
            F_L, F_R = div_a_grad_perp_flows(J, g11, dx, dy, dz, a, f, ben_strict)
            F_heat_L += F_L
            F_heat_R += F_R
    F_part_L, F_part_R = div_a_grad_perp_flows(J, g11, dx, dy, dz,
                                               coef("D", ion_name), Ni, ben_strict)

    core = core_y_mask(meta.get("snowflake_type", ""), ny, j11, j21, j12, j22)

    # ── Core volumetric source (CLAUDE_BEN_FLUXES.md §2.4) ────────────────────
    P_core = 1.5 * sum(vol(xy(f"P{sp}_src")) for sp in ("e", ion_name)
                       if has(f"P{sp}_src"))

    # ── Sheath at every target ────────────────────────────────────────────────
    sec = "sheath_boundary_simple"
    gamma_e = _opt(opts, sec, "gamma_e", 3.5) if gamma_e is None else gamma_e
    gamma_i = _opt(opts, sec, "gamma_i", 3.5) if gamma_i is None else gamma_i
    sheath_kw = dict(AA=ion_AA, Zi=ion_Z,
                     sheath_ion_polytropic=_opt(opts, sec, "sheath_ion_polytropic", 1.0),
                     density_mode=_opt(opts, sec, "density_boundary_mode", 1),
                     temperature_mode=_opt(opts, sec, "temperature_boundary_mode", 1),
                     ben_strict=ben_strict)
    targets = sheath_targets(ny_inner, ny, single_null)
    slots = [0, 3] if single_null else [0, 1, 2, 3]
    slot_power = [np.zeros_like(Ne[:, 0]) for _ in range(4)]
    slot_area = [np.zeros_like(Ne[:, 0]) for _ in range(4)]
    P_sheath_e, P_sheath_i, G_sheath = [], [], []
    for (y, side), s in zip(targets, slots):
        pe, pi, pn = sheath_target_flows(Ne, Ni, Te, Ti, Vi, area[:, y], y, side,
                                         gamma_e, gamma_i, **sheath_kw)
        slot_power[s] = pe + pi
        slot_area[s] = area[:, y]
        P_sheath_e.append(float(pe[xs].sum()))
        P_sheath_i.append(float(pi[xs].sum()))
        G_sheath.append(float(pn[xs].sum()))

    # ── Radiation (×1.5 and no impurity only under ben_strict, §3 item 6) ─────
    rad_names = [f"R{ion_name}_ex", f"R{ion_name}_rec"] + ([] if ben_strict else ["Rc"])
    rad_factor = 1.5 if ben_strict else 1.0
    P_rad_parts = {n: -rad_factor * vol(xy(n)) for n in rad_names if has(n)}

    # ── Neutral boundary energy sources (§3 item 9) ───────────────────────────
    P_neutral_parts = {}
    if not ben_strict:
        for suffix in ("target_recycle", "wall_refl", "target_refl", "pump"):
            n = f"E{neutral}_{suffix}"
            if has(n):
                P_neutral_parts[n] = vol(xy(n))

    # ── Rate of change of stored energy (§3 item 7) ───────────────────────────
    species = ("e", ion_name, neutral)
    dEdt_int = 1.5 * sum(vol(xy(f"ddt(P{sp})")) for sp in species if has(f"ddt(P{sp})"))
    dEdt_kin = 0.0
    for sp in species:
        if all(has(n) for n in (f"ddt(NV{sp})", f"ddt(N{sp})", f"NV{sp}", f"V{sp}", f"N{sp}")):
            V = xy(f"V{sp}")
            dEdt_kin += vol(V * xy(f"ddt(NV{sp})")
                            - 0.5 * xy(f"NV{sp}") * V * xy(f"ddt(N{sp})") / xy(f"N{sp}"))

    P_x_in_core = float(F_heat_L[xin, core].sum())
    P_x_in_pf = float(F_heat_L[xin, ~core].sum())
    P_x_out = float(F_heat_R[xout].sum())
    P_rad = sum(P_rad_parts.values())
    P_neutral = sum(P_neutral_parts.values())
    P_net = (P_core + P_x_in_core + P_x_in_pf - P_x_out
             - sum(P_sheath_e) - sum(P_sheath_i) - P_rad + P_neutral)

    # ── Particle balance ──────────────────────────────────────────────────────
    G_x_in_core = float(F_part_L[xin, core].sum())
    G_x_in_pf = float(F_part_L[xin, ~core].sum())
    G_x_out = float(F_part_R[xout].sum())
    G_sum = sum(G_sheath)
    recycle = f"S{neutral}_target_recycle"

    return dict(
        strict=ben_strict, gamma_e=gamma_e, gamma_i=gamma_i,
        targets=targets, slot_power=slot_power, slot_area=slot_area,
        P_core=P_core, P_x_in_core=P_x_in_core, P_x_in_pf=P_x_in_pf, P_x_out=P_x_out,
        P_sheath_e=P_sheath_e, P_sheath_i=P_sheath_i,
        P_rad=P_rad, P_rad_parts=P_rad_parts,
        P_neutral=P_neutral, P_neutral_parts=P_neutral_parts,
        P_net=P_net, dEdt_int=dEdt_int, dEdt_kin=dEdt_kin,
        residual=P_net - dEdt_int - dEdt_kin,
        G_x_in_core=G_x_in_core, G_x_in_pf=G_x_in_pf, G_x_out=G_x_out, G_sheath=G_sheath,
        f_recycle_ben=1.0 - (G_x_in_core + G_x_in_pf - G_x_out) / G_sum if G_sum else np.nan,
        f_recycle_direct=vol(xy(recycle)) / G_sum if (has(recycle) and G_sum) else np.nan,
        G_iz=vol(xy(f"S{ion_name}_iz")) if has(f"S{ion_name}_iz") else np.nan,
        G_rec=vol(xy(f"S{ion_name}_rec")) if has(f"S{ion_name}_rec") else np.nan,
    )


# ── Plotting helpers ────────────────────────────────────────────────────────────

def plot_heat_flux(datasets, labels, colors, lstyles, q_key, ylabel, title, out_path, xlim, ylim):
    """Generic 2×2 heat flux figure (one panel per target)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ti, (ax, tlabel) in enumerate(zip(axes, TARGET_LABELS)):
        for data, label, color, ls in zip(datasets, labels, colors, lstyles):
            d    = data["d_sep"]
            q    = data[q_key][ti]
            MXG  = data["MXG"]
            frac = data["fractions"][ti]
            ax.plot(d[MXG:-MXG], q[MXG:-MXG], label=f"{label}  ({frac:.1f}%)",
                    color=color, linestyle=ls, linewidth=1.8)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel(r"$(R - R_\mathrm{sep})\,/\,\rho_{s0}$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(tlabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}.pdf / .png")


# ── Grid plotting (independent feature — still works) ────────────────────────────

def plot_grids(cases, out_path):
    """Plot each provided SF grid side by side (grids only, no separatrices)."""
    if vig is None:
        print(f"--plot_grids: could not import visualize_in_grid from {_VIG_DIR!r}; "
              "skipping grid plot.")
        return
    panels, seen = [], set()
    for c in cases:
        if c["grid"] not in seen:
            seen.add(c["grid"])
            panels.append((c["grid"], c["label"]))
    if not panels:
        print("--plot_grids: no grids available to plot.")
        return
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 9), squeeze=False)
    axes = axes[0]
    for ax, (grid, title) in zip(axes, panels):
        try:
            ds   = vig.load_grid(grid)
            topo = vig.get_topology_indices(ds)
            vig.plot_grid_only(ds, topo, ax, edgecolor="steelblue", linewidth=0.3)
            vig.plot_wall(ds, ax)
            vig.set_axis_style(ax, ds, title=title)
            ds.close()
        except Exception as e:
            print(f"--plot_grids: could not plot {title} ({grid}): {e}")
            ax.set_title(f"{title}\n(failed)")
            ax.set_axis_off()
    fig.suptitle("SF divertor grids", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}.pdf / .png")


# ── Flux-surface / equilibrium profiles (Giacomin Fig. 3 style) ─────────────────
# These are purely EQUILIBRIUM quantities read from the grid file — independent of
# the simulation data. rho_s0 only enters to normalise the connection-length axes.

def _read_rho_s0(sim_path):
    """Read the SI length scale rho_s0 [m] straight from the first dump file."""
    f = sorted(glob.glob(_dump_pattern(sim_path)))[0]
    ds = netCDF4.Dataset(f)
    val = float(np.array(ds["rho_s0"][...]).flatten()[0])
    ds.close()
    return val


def compute_flux_surface_profiles(grid_file, rho_s0, mxg=2):
    """
    Equilibrium profiles for the Giacomin Fig. 3 panels, from the grid file.

    (a) Safety factor  q(rho_N)   — closed surfaces only.
        q = |ShiftAngle| / 2*pi. ShiftAngle is the toroidal twist over one full
        poloidal circuit of a closed flux surface, computed topology-correctly by
        the grid generator (validated == direct integral of Bt*hthe/(R*Bp) over
        the inner+outer core loop). This sidesteps the non-standard snowflake
        jyseps ordering.
    (b) Magnetic shear  s(rho_N) = (rho_N / q) dq/drho_N.
    (c) LFS connection length  L_par/rho_s0  vs  (R - R_sep)/rho_s0.
        L_par = integral of (B/Bp)*hthe*dy from the outboard (max-R) midplane to
        the LFS target (poloidal index ny_inner-1), on SOL surfaces. Diverges where
        Bp -> 0 (a secondary X-point on the LFS leg).

    rho_N = sqrt((psi - psi_0)/(psi_sep - psi_0)), psi at the outboard midplane,
    psi_0 at the innermost core surface, psi_sep at ixseps1.

    Returns dict: rho_N, q, s (core arrays); L_dist, L_par (SOL arrays).
    """
    g = netCDF4.Dataset(grid_file)
    gi = lambda k: int(np.array(g[k][...]).flatten()[0])
    ix1, ny, nyi, nx = gi("ixseps1"), gi("ny"), gi("ny_inner"), gi("nx")
    R   = np.array(g["Rxy"][:]);  psi = np.array(g["psixy"][:])
    Bp  = np.array(g["Bpxy"][:]); B   = np.array(g["Bxy"][:])
    hthe = np.array(g["hthe"][:]); dy = np.array(g["dy"][:])
    SA  = np.array(g["ShiftAngle"][:]).flatten()
    g.close()

    ymid = int(np.argmax(R[ix1, :]))            # outboard (max-R) midplane

    # ── (a) safety factor on closed core surfaces (x = mxg .. ixseps1-1) ─────
    xc = np.arange(mxg, ix1)
    q  = np.abs(SA[xc]) / (2.0 * np.pi)
    psi0, psis = psi[mxg, ymid], psi[ix1, ymid]
    rho_N = np.sqrt(np.clip((psi[xc, ymid] - psi0) / (psis - psi0), 0.0, None))

    # ── (b) magnetic shear s = (rho_N/q) dq/drho_N ───────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        s = rho_N / q * np.gradient(q, rho_N)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    # ── (c) LFS connection length on SOL surfaces (x = ixseps1 .. nx-mxg-1) ──
    xs = np.arange(ix1, nx - mxg)
    ys = np.arange(ymid, nyi)                   # midplane -> LFS target
    L_par  = np.array([np.sum((B[x, ys] / Bp[x, ys]) * hthe[x, ys] * dy[x, ys])
                       for x in xs]) / rho_s0
    L_dist = (R[xs, ymid] - R[ix1, ymid]) / rho_s0

    return dict(rho_N=rho_N, q=q, s=s, L_dist=L_dist, L_par=L_par)


def plot_flux_surface_figure(profiles, labels, colors, lstyles, out_path):
    """Three-panel figure: (a) <q>, (b) <s> vs rho_N; (c) L_par vs distance."""
    fig = plt.figure(figsize=(11, 9))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    ax_q = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_L = fig.add_subplot(gs[1, :])

    for f, label, color, ls in zip(profiles, labels, colors, lstyles):
        ax_q.plot(f["rho_N"], f["q"], color=color, linestyle=ls, lw=1.8, label=label)
        ax_s.plot(f["rho_N"], f["s"], color=color, linestyle=ls, lw=1.8, label=label)
        ax_L.plot(f["L_dist"], f["L_par"], color=color, linestyle=ls, lw=1.8, label=label)

    ax_q.set_xlabel(r"$\rho_N$");           ax_q.set_ylabel(r"$\langle q\rangle_\psi$")
    ax_s.set_xlabel(r"$\rho_N$");           ax_s.set_ylabel(r"$\langle s\rangle_\psi$")
    ax_L.set_xlabel(r"Distance from the primary separatrix $(R - R_\mathrm{sep})/\rho_{s0}$")
    ax_L.set_ylabel(r"Connection length $L_\parallel/\rho_{s0}$")
    ax_q.set_title("(a) Safety factor", fontsize=11)
    ax_s.set_title("(b) Magnetic shear", fontsize=11)
    ax_L.set_title("(c) Low-field side connection length", fontsize=11)
    for ax in (ax_q, ax_s, ax_L):
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}.pdf / .png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SI-clean (xHermes) SF target heat-flux analysis. Target power "
                    "from energy-flow diagnostics (A, default), the analytic "
                    "sheath formula (C, --analytic_power) or Ben's flux and "
                    "balance calculation (D, --ben_example_fluxes)."
    )
    parser.add_argument("sims_dir",
                        help="Folder containing the SF<angle>/SF-ideal/SN simulation folders")
    parser.add_argument("grid_dir",
                        help="Folder containing the grid files (*sf<angle>*.nc, *sf_pro_max*.nc, *lsn*.nc)")
    parser.add_argument("--only", nargs="+", default=None, metavar="CASE",
                        help="Only load these sim folders, e.g. --only SF15 SF45 SN")
    parser.add_argument("--twindow", type=int, default=20,
                        help="Number of final timesteps to time-average (default: 20)")
    parser.add_argument("--analytic_power", action="store_true", default=False,
                        help="Use the analytic sheath flux (approach C) instead of the "
                             "energy-flow diagnostics (approach A, default).")
    parser.add_argument("--ben_example_fluxes", action="store_true", default=False,
                        help="Approach D (Ben_example_fluxes): sheath power from "
                             "analyse_fluxes.py adapted to SF, at the last clean step, "
                             "plus the power/particle balance table (CLAUDE_BEN_FLUXES.md).")
    parser.add_argument("--ben_strict", action="store_true", default=False,
                        help="With --ben_example_fluxes: use Ben's formulas exactly instead "
                             "of the Hermes-source ones (CLAUDE_BEN_FLUXES.md §3).")
    parser.add_argument("--gamma_e", type=float, default=None,
                        help="Electron sheath heat transmission coefficient. C: default 3.5; "
                             "D: default from BOUT.settings [sheath_boundary_simple].")
    parser.add_argument("--gamma_i", type=float, default=None,
                        help="Ion sheath heat transmission coefficient. C: default 3.5; "
                             "D: default from BOUT.settings [sheath_boundary_simple].")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"),
                        help="x-axis limits in rho_s0 units, e.g. --xlim -5 20")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                        help="y-axis limits [MW/m^2] for the SI heat-flux plot")
    parser.add_argument("--ylim_norm", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                        help="y-axis limits for the normalised heat-flux plot")
    parser.add_argument("--plot_grids", action="store_true", default=False,
                        help="Plot each case's grid side by side (grids only).")
    parser.add_argument("--flux_profiles", action="store_true", default=False,
                        help="Also produce the equilibrium 3-panel figure: safety "
                             "factor q(rho_N), magnetic shear s(rho_N), and LFS "
                             "connection length L_par/rho_s0 (Giacomin Fig. 3 style).")
    parser.add_argument("--output", type=str, default=None, metavar="TAG",
                        help="Extra tag appended to every saved figure name, "
                             "e.g. --output No-drifts -> SF_heat_flux_targets_A_No-drifts.pdf")
    args = parser.parse_args()
    if args.ben_example_fluxes and args.analytic_power:
        parser.error("--ben_example_fluxes and --analytic_power cannot be combined.")
    if args.ben_strict and not args.ben_example_fluxes:
        parser.error("--ben_strict needs --ben_example_fluxes.")

    for d in (args.sims_dir, args.grid_dir):
        if not os.path.isdir(d):
            print(f"Error: '{d}' is not a directory.")
            sys.exit(1)
    cases = discover_cases(args.sims_dir, args.grid_dir, only=args.only)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    approach_tag = ("D_strict" if args.ben_strict else "D" if args.ben_example_fluxes
                    else "C" if args.analytic_power else "A")
    out_sfx = f"_{args.output}" if args.output else ""

    if args.plot_grids:
        plot_grids(cases, os.path.join(script_dir, f"SF_grids{out_sfx}"))

    datasets, labels, colors, lstyles = [], [], [], []
    for c in cases:
        data = load_simulation(c["path"], c["grid"], t_window=args.twindow,
                               ben_example=args.ben_example_fluxes,
                               ben_strict=args.ben_strict,
                               gamma_e=args.gamma_e, gamma_i=args.gamma_i)
        compute_target_power(data, analytic=args.analytic_power,
                             gamma_e=3.5 if args.gamma_e is None else args.gamma_e,
                             gamma_i=3.5 if args.gamma_i is None else args.gamma_i,
                             ben=args.ben_example_fluxes)
        datasets.append(data)
        labels.append(c["label"])
        colors.append(c["color"])
        lstyles.append(c["ls"])

    if not datasets:
        if args.plot_grids:
            plt.show()
            return
        print("No simulations found.")
        sys.exit(1)

    print_si_sanity(datasets, labels)
    print(f"\nTopology (xbout): {datasets[0].get('topology')}    "
          f"Power approach: {datasets[0]['approach']}")
    print_power_table(datasets, labels)
    if args.ben_example_fluxes:
        print_balance_table(datasets, labels)

    # ── Figure 1: SI units [MW/m^2] ──────────────────────────────────────────
    plot_heat_flux(
        datasets, labels, colors, lstyles,
        q_key   = "q",
        ylabel  = r"$q_\parallel\;[\mathrm{MW\,m}^{-2}]$",
        title   = r"Total parallel heat flux ($q_e + q_i$) at divertor targets",
        out_path= os.path.join(script_dir, f"SF_heat_flux_targets_{approach_tag}{out_sfx}"),
        xlim    = args.xlim, ylim = args.ylim,
    )

    # ── Figure 2: normalised by q_0 ──────────────────────────────────────────
    plot_heat_flux(
        datasets, labels, colors, lstyles,
        q_key   = "q_norm",
        ylabel  = r"$q_\parallel\,/\,q_0$",
        title   = r"Normalised parallel heat flux ($q_e + q_i$) / $q_0$ at divertor targets",
        out_path= os.path.join(script_dir, f"SF_heat_flux_targets_normalised_{approach_tag}{out_sfx}"),
        xlim    = args.xlim, ylim = args.ylim_norm,
    )

    # ── Figure 3: power fraction bar chart ───────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    x     = np.arange(4)
    width = 0.8 / len(datasets)
    for i, (data, label, color) in enumerate(zip(datasets, labels, colors)):
        offset = (i - len(datasets) / 2 + 0.5) * width
        bars = ax3.bar(x + offset, data["fractions"], width,
                       label=label, color=color, alpha=0.85)
        for bar, frac in zip(bars, data["fractions"]):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{frac:.1f}%", ha="center", va="bottom", fontsize=7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(TARGET_SHORT)
    ax3.set_ylabel("Power fraction [%]")
    ax3.set_title("Fraction of total target power at each strike point")
    ax3.legend(fontsize=9)
    ax3.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    out_frac = os.path.join(script_dir, f"SF_power_fractions_{approach_tag}{out_sfx}")
    plt.savefig(out_frac + ".pdf", bbox_inches="tight")
    plt.savefig(out_frac + ".png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_frac}.pdf / .png")

    # ── Optional Figure 4: equilibrium flux-surface profiles (Giacomin Fig. 3) ─
    if args.flux_profiles:
        profiles, fp_labels, fp_colors, fp_lstyles = [], [], [], []
        for c in cases:
            try:
                rho_s0 = _read_rho_s0(c["path"])
                profiles.append(compute_flux_surface_profiles(c["grid"], rho_s0))
                fp_labels.append(c["label"]); fp_colors.append(c["color"]); fp_lstyles.append(c["ls"])
            except Exception as e:
                print(f"--flux_profiles: could not build profiles for {c['label']}: {e}")
        if profiles:
            plot_flux_surface_figure(profiles, fp_labels, fp_colors, fp_lstyles,
                                     os.path.join(script_dir, f"SF_flux_surface_profiles{out_sfx}"))

    plt.show()


if __name__ == "__main__":
    main()
