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
    SE (SP3) : theta = ny_inner + 1
    SW (SP4) : theta = ny - 1
"""

import sys
import os
import glob
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

GEOM_KEYS     = ["sf_plus",  "hfs_sfm",  "lfs_sfm",  "ideal_sf", "sn",         "sf45",      "sf135"]
GEOM_LABELS   = ["SF+",      "HFS SF−",  "LFS SF−",  "Ideal SF", "SN",         "SF45",      "SF135"]
GEOM_METAVARS = ["SF+",      "HFS_SF-",  "LFS_SF-",  "IdealSF",  "SN",         "SF45",      "SF135"]
COLORS        = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
LINESTYLES    = ["-",        "--",       "-.",       ":",        (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

TARGET_LABELS = [
    "SP1 – NW target",
    "SP2 – NE target",
    "SP3 – SE target",
    "SP4 – SW target",
]
TARGET_SHORT = ["SP1 (NW)", "SP2 (NE)", "SP3 (SE)", "SP4 (SW)"]


# ── Time handling ──────────────────────────────────────────────────────────────

def _common_nt(sim_path):
    """
    Largest number of output steps present in *every* BOUT.dmp file.

    Per-rank dump files (esp. with the PETSc solver) need not flush the same
    number of steps; reading a step that a short rank never wrote crashes the
    loader. We clip the time window to the last common step. Returns
    ``(min_t, max_t, nfiles)``.
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

def load_simulation(sim_path, grid_file, t_window=20):
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
        target_y   : list of 4 poloidal indices [NW, NE, SE, SW] (no-guard frame)
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
    """
    print(f"  Loading {sim_path} ...")

    # Clip to the last common step across ranks, then open in SI (unnormalise=True).
    min_t, max_t, nfiles = _common_nt(sim_path)
    if min_t == 0:
        raise ValueError(f"{sim_path}: a dump file has no time steps; cannot average.")
    if min_t != max_t:
        print(f"    Time-step mismatch across {nfiles} dump files "
              f"(t lengths {min_t}..{max_t}); clipping to t index {min_t - 1}.")

    # Clean toroidal geometry from the grid's psixy/Rxy. keep_yboundaries=False
    # (default) so the poloidal frame matches the grid (no y-guards). The corner
    # mesh (rm/zm) is kept aside for real-space plotting.
    field_grid, corner_mesh = _open_clean_grid(grid_file)
    ds = xhermes.open(sim_path, geometry="toroidal",
                      gridfilepath=field_grid, info=False)
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

    # ── Slice the building blocks at each target poloidal index ──────────────
    target_y = [0, ny_inner - 1, ny_inner + 1, ny - 1]

    def col(arr2d, yi):
        return arr2d[:, yi]

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

def compute_target_power(data, analytic=False, gamma_e=3.5, gamma_i=3.5):
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

    Both routes write the same downstream keys into `data`:
        q        : list of 4 (nx,) arrays, parallel heat flux  [MW/m^2]
        q_norm   : list of 4 (nx,) arrays, q / q_0             [dimensionless]
        P_targets: list of 4 floats, integrated power per target [MW]
        P_total  : total power across the 4 targets             [MW]
        fractions: list of 4 floats, power fraction per target  [%]
        approach : "A (energy-flow)" or "C (analytic sheath)"
    """
    g = data["MXG"]
    sl = slice(g, -g if g else None)          # interior radial cells
    q_0 = data["q_0"]                          # [W/m^2]

    q_list, P_list = [], []
    for i in range(4):
        area = data["area"][i]
        if analytic:
            mi  = data["ion_AA"] * MP
            Zi  = data["ion_Z"]
            TeJ = data["Te"][i] * QE
            TiJ = data["Ti"][i] * QE
            Cs  = np.sqrt(np.clip((Zi * TeJ + TiJ) / mi, 0.0, None))   # [m/s]
            q   = gamma_e * data["Ne"][i] * TeJ * Cs \
                + gamma_i * data["Ni"][i] * TiJ * Cs                   # [W/m^2]
        else:
            power_cell = np.abs(data["efe_power"][i]) + np.abs(data["ion_power"][i])
            q = power_cell / area                                     # [W/m^2]
        q_list.append(q)
        P_list.append(float(np.sum((q * area)[sl])))                  # [W]

    P_total = sum(P_list)
    data["q"]        = [q / 1e6 for q in q_list]                      # MW/m^2
    data["q_norm"]   = [q / q_0 for q in q_list]                      # dimensionless
    data["P_targets"] = [P / 1e6 for P in P_list]                     # MW
    data["P_total"]  = P_total / 1e6                                  # MW
    data["fractions"] = [100.0 * P / P_total if P_total else 0.0 for P in P_list]
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

GRID_PLOT_ORDER = [
    ("hfs_sfm",  "HFS-SF"),
    ("sf45",     "SF45"),
    ("sf_plus",  "SF75"),
    ("ideal_sf", "Ideal-SF"),
    ("sf135",    "SF135"),
    ("lfs_sfm",  "LFS-SF"),
    ("sn",       "LSN"),
]


def _resolve_grid_for_plot(args, key):
    grid = getattr(args, f"grid_{key}")
    if grid is None and getattr(args, key) is not None:
        grid = args.grid
    return grid


def plot_grids(args, out_path):
    """Plot each provided SF grid side by side (grids only, no separatrices)."""
    if vig is None:
        print(f"--plot_grids: could not import visualize_in_grid from {_VIG_DIR!r}; "
              "skipping grid plot.")
        return
    panels = []
    for key, title in GRID_PLOT_ORDER:
        grid = _resolve_grid_for_plot(args, key)
        if grid is None:
            continue
        if not os.path.isfile(grid):
            print(f"--plot_grids: grid file '{grid}' for {title} not found – skipping.")
            continue
        panels.append((grid, title))
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
    f = sorted(glob.glob(os.path.join(sim_path, "BOUT.dmp.*.nc")))[0]
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
                    "from energy-flow diagnostics (A, default) or the analytic "
                    "sheath formula (C, --analytic_power)."
    )
    parser.add_argument("--grid", default=None, metavar="GRID_FILE",
                        help="Fallback BOUT++ grid (.nc) for geometries without a --grid-<name>")
    for key, label, metavar in zip(GEOM_KEYS, GEOM_LABELS, GEOM_METAVARS):
        parser.add_argument(key, nargs="?", default=None, metavar=metavar,
                            help=f"{label} simulation path")
        parser.add_argument(f"--grid-{key}", dest=f"grid_{key}", default=None,
                            metavar="GRID_FILE", help=f"Grid file for {label}")
    parser.add_argument("--twindow", type=int, default=20,
                        help="Number of final timesteps to time-average (default: 20)")
    parser.add_argument("--analytic_power", action="store_true", default=False,
                        help="Use the analytic sheath flux (approach C) instead of the "
                             "energy-flow diagnostics (approach A, default).")
    parser.add_argument("--gamma_e", type=float, default=3.5,
                        help="Electron sheath heat transmission coefficient for C (default: 3.5)")
    parser.add_argument("--gamma_i", type=float, default=3.5,
                        help="Ion sheath heat transmission coefficient for C (default: 3.5)")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"),
                        help="x-axis limits in rho_s0 units, e.g. --xlim -5 20")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                        help="y-axis limits [MW/m^2] for the SI heat-flux plot")
    parser.add_argument("--ylim_norm", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                        help="y-axis limits for the normalised heat-flux plot")
    parser.add_argument("--plot_grids", action="store_true", default=False,
                        help="Plot each supplied grid side by side (grids only).")
    parser.add_argument("--flux_profiles", action="store_true", default=False,
                        help="Also produce the equilibrium 3-panel figure: safety "
                             "factor q(rho_N), magnetic shear s(rho_N), and LFS "
                             "connection length L_par/rho_s0 (Giacomin Fig. 3 style).")
    args = parser.parse_args()

    if args.grid is not None and not os.path.isfile(args.grid):
        print(f"Error: fallback grid file '{args.grid}' not found.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.plot_grids:
        plot_grids(args, os.path.join(script_dir, "SF_grids"))

    datasets, labels, colors, lstyles = [], [], [], []
    for key, label, color, ls in zip(GEOM_KEYS, GEOM_LABELS, COLORS, LINESTYLES):
        path = getattr(args, key)
        if path is None:
            continue
        if not os.path.isdir(path):
            print(f"Warning: '{path}' is not a directory – skipping {label}.")
            continue
        grid = getattr(args, f"grid_{key}") or args.grid
        if grid is None:
            print(f"Warning: no grid for {label} (use --grid-{key} or --grid) – skipping.")
            continue
        if not os.path.isfile(grid):
            print(f"Warning: grid file '{grid}' for {label} not found – skipping.")
            continue
        data = load_simulation(path, grid, t_window=args.twindow)
        compute_target_power(data, analytic=args.analytic_power,
                             gamma_e=args.gamma_e, gamma_i=args.gamma_i)
        datasets.append(data)
        labels.append(label)
        colors.append(color)
        lstyles.append(ls)

    if not datasets:
        if args.plot_grids:
            plt.show()
            return
        print("No valid simulation paths provided.")
        sys.exit(1)

    print_si_sanity(datasets, labels)
    print(f"\nTopology (xbout): {datasets[0].get('topology')}    "
          f"Power approach: {datasets[0]['approach']}")
    print_power_table(datasets, labels)

    # ── Figure 1: SI units [MW/m^2] ──────────────────────────────────────────
    plot_heat_flux(
        datasets, labels, colors, lstyles,
        q_key   = "q",
        ylabel  = r"$q_\parallel\;[\mathrm{MW\,m}^{-2}]$",
        title   = r"Total parallel heat flux ($q_e + q_i$) at divertor targets",
        out_path= os.path.join(script_dir, "SF_heat_flux_targets"),
        xlim    = args.xlim, ylim = args.ylim,
    )

    # ── Figure 2: normalised by q_0 ──────────────────────────────────────────
    plot_heat_flux(
        datasets, labels, colors, lstyles,
        q_key   = "q_norm",
        ylabel  = r"$q_\parallel\,/\,q_0$",
        title   = r"Normalised parallel heat flux ($q_e + q_i$) / $q_0$ at divertor targets",
        out_path= os.path.join(script_dir, "SF_heat_flux_targets_normalised"),
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
    out_frac = os.path.join(script_dir, "SF_power_fractions")
    plt.savefig(out_frac + ".pdf", bbox_inches="tight")
    plt.savefig(out_frac + ".png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_frac}.pdf / .png")

    # ── Optional Figure 4: equilibrium flux-surface profiles (Giacomin Fig. 3) ─
    if args.flux_profiles:
        profiles, fp_labels, fp_colors, fp_lstyles = [], [], [], []
        for key, label, color, ls in zip(GEOM_KEYS, GEOM_LABELS, COLORS, LINESTYLES):
            path = getattr(args, key)
            if path is None or not os.path.isdir(path):
                continue
            grid = getattr(args, f"grid_{key}") or args.grid
            if grid is None or not os.path.isfile(grid):
                continue
            try:
                rho_s0 = _read_rho_s0(path)
                profiles.append(compute_flux_surface_profiles(grid, rho_s0))
                fp_labels.append(label); fp_colors.append(color); fp_lstyles.append(ls)
            except Exception as e:
                print(f"--flux_profiles: could not build profiles for {label}: {e}")
        if profiles:
            plot_flux_surface_figure(profiles, fp_labels, fp_colors, fp_lstyles,
                                     os.path.join(script_dir, "SF_flux_surface_profiles"))

    plt.show()


if __name__ == "__main__":
    main()
