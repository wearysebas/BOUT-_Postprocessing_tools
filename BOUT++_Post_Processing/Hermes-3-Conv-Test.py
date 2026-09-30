#!/usr/bin/env python3
"""
Hermes-3 convergence-timescale visualization on the gridue polygon mesh.

This is a sibling of ``Hermes-3_postprocessing.py``. The data loading and the
polygon plotting / animation are identical; the difference is *what* gets
plotted. Instead of showing a raw field, this script shows the local
convergence timescale of an evolving quantity::

    var / (ddt(var) * Omega_ci)

i.e. the (physical, seconds) e-folding time over which ``var`` is still
changing. ``ddt(var)`` is the matching time-derivative field that BOUT++/Hermes-3
writes into the dump (literally the variable named ``ddt(<var>)``), and
``Omega_ci`` is the ion-cyclotron normalisation frequency from the run, which
converts the normalised rate into physical units. Large magnitude => slowly
evolving (well converged); small magnitude => still moving fast.

``var`` is restricted to the evolving pressures / densities / momenta:
    Pe, Pd+, Pd, Nd+, Nd, NVe, NVd+, NVd

The denominator carries a tiny additive guard (``ddt(var) * Omega_ci + 1e-10``)
so cells where the time derivative has gone to zero (fully converged) give a
large finite timescale instead of dividing by zero.

Usage:
    python Hermes-3-Conv-Test.py <grid_file> <sim_results_dir> [options]

Example:
    python Hermes-3-Conv-Test.py \
        MAST-U_SF75.grd.nc Bender_mesh_Hermes/examples/... \
        --var Pe --time -1 --cmap RdBu_r
"""

import glob
import os
import re

import numpy as np
import xarray as xr
import xbout
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


# Variables that this convergence test accepts. Each must have a matching
# ``ddt(<var>)`` field in the dump files.
ALLOWED_VARS = ["Pe", "Pd+", "Pd", "Nd+", "Nd", "NVe", "NVd+", "NVd"]


def _proc_index(path):
    """Sort key: numeric processor index from a BOUT.dmp.<n>.nc filename."""
    m = re.search(r"BOUT\.dmp\.(\d+)\.nc$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def open_aligned_boutdataset(datapath, **kwargs):
    """
    Load a Hermes-3 / BOUT++ run whose per-processor dump files may contain
    different numbers of time steps.

    Opens each ``BOUT.dmp.*.nc`` separately, truncates every file to the last
    time index present in *all* of them, and hands the aligned list to xbout so
    its normal spatial combine / guard-cell handling applies unchanged. (See
    ``Hermes-3_postprocessing.py`` for the full rationale.)
    """
    # Accept a directory, a glob pattern, or a path to a single representative file.
    if os.path.isdir(datapath):
        pattern = os.path.join(datapath, "BOUT.dmp.*.nc")
    else:
        pattern = datapath

    files = sorted(glob.glob(pattern), key=_proc_index)
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp files matched: {pattern!r}")

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
            f"[Hermes-3-Conv-Test] Time-step mismatch across dump files: "
            f"lengths range {min_t}..{max_t} "
            f"({n_short}/{len(tlens)} ranks short of the maximum). "
            f"Truncating all files to the last common step (t index 0..{min_t - 1})."
        )
    else:
        print(
            f"[Hermes-3-Conv-Test] All {len(files)} dump files agree on "
            f"{min_t} time steps; no truncation needed."
        )

    datasets = [ds.isel(t=slice(0, min_t)) for ds in datasets]

    return xbout.open_boutdataset(datapath=datasets, **kwargs)


def get_omega_ci(sim):
    """
    Pull the ion-cyclotron normalisation frequency ``Omega_ci`` from a loaded
    run, whether it was stored as a (scalar) data variable, in xbout's
    ``metadata`` dict, or in the dataset ``attrs``.
    """
    if "Omega_ci" in sim.data_vars:
        return float(np.asarray(sim["Omega_ci"].values).flat[0])
    meta = getattr(sim, "metadata", {}) or {}
    if "Omega_ci" in meta:
        return float(meta["Omega_ci"])
    if "Omega_ci" in getattr(sim, "attrs", {}):
        return float(sim.attrs["Omega_ci"])
    raise KeyError(
        "Could not find 'Omega_ci' in the simulation results "
        "(checked data_vars, metadata and attrs)."
    )


# Small additive guard on the denominator so fully converged cells
# (ddt(var) -> 0) give a large finite timescale instead of inf / a crash.
_DENOM_EPS = 1e-10


def convergence_field(sim, var, MXG, zindex, omega_ci, time=None,
                      plot_change=False):
    """
    Compute the convergence timescale ``var / (ddt(var) * Omega_ci + eps)``, or
    its inverse, the fractional rate of change ``(ddt(var) * Omega_ci) / (var +
    eps)`` when ``plot_change`` is True.

    The x guard cells are stripped (``MXG``) and the toroidal index ``zindex`` is
    selected, exactly as in the raw-field plotter. A tiny ``eps`` (``1e-10``) is
    added to the denominator so the division never blows up: by default this
    guards ``ddt(var)`` going to zero (fully converged cells -> large finite
    timescale); with ``plot_change`` it guards ``var`` going to zero (empty cells
    -> finite rate).

    Parameters
    ----------
    time : int or None
        If an int, return a single ``(bout_nx, bout_ny)`` frame at that time
        index. If None, return the whole ``(nt, bout_nx, bout_ny)`` series (for
        animation).
    plot_change : bool
        If True, return the rate of change ``(ddt(var) * Omega_ci) / var`` [1/s]
        instead of the timescale ``var / (ddt(var) * Omega_ci)`` [s].
    """
    ddt_name = f"ddt({var})"
    if var not in sim.data_vars:
        raise KeyError(f"Variable {var!r} not found in the simulation results.")
    if ddt_name not in sim.data_vars:
        raise KeyError(
            f"Time-derivative field {ddt_name!r} not found in the simulation "
            f"results; cannot compute the convergence ratio for {var!r}."
        )

    if time is None:
        v = sim[var].values[:, MXG:-MXG, :, zindex]
        d = sim[ddt_name].values[:, MXG:-MXG, :, zindex]
    else:
        v = sim[var].values[time, MXG:-MXG, :, zindex]
        d = sim[ddt_name].values[time, MXG:-MXG, :, zindex]

    if plot_change:
        return np.abs((d * omega_ci) / (v + _DENOM_EPS))
    return np.abs(v / (d * omega_ci + _DENOM_EPS))


def _build_patches(rm, zm, ny_inner, bout_nx, bout_ny, bridge_cut=False):
    """
    Build the polygon patches and their (bout_i, bout_j) data indices by
    iterating over the raw gridue cells. Geometry is time-independent, so this
    is computed once and reused for every animation frame.

    Gridue rm/zm have shape (Nx_gridue, Ny_gridue, 5): dim0 poloidal (BOUT y),
    dim1 radial (BOUT x), dim2 corner (0=center, 1-4=corners).

    ``bridge_cut`` closes the two poloidal guard columns at the ny_inner branch
    cut by colouring them from their neighbouring interior cells. Only pass True
    for a single null (``jyseps2_1 == jyseps1_2``), where that cut is a
    continuous top-of-domain location; for a double null / snowflake it is a
    real inner/outer target break and must stay open.
    """
    Nx_gridue = rm.shape[0]
    Ny_gridue = rm.shape[1]

    patches = []
    ii = []
    jj = []
    idx = [np.array([1, 2, 4, 3, 1])]

    for i in range(Nx_gridue):
        for j in range(Ny_gridue):
            # Radial: skip boundary cells (j=0 and j=Ny-1)
            if j == 0 or j == Ny_gridue - 1:
                continue
            bout_i = j + 1

            # Poloidal: skip boundary cells (i=0 and i=Nx-1) and guard cells
            if i == 0 or i == Nx_gridue - 1:
                continue
            post_step1 = i - 1
            if post_step1 < ny_inner:
                bout_j = post_step1
            elif post_step1 == ny_inner or post_step1 == ny_inner + 1:
                continue  # Guard cell at ny_inner boundary
            else:
                bout_j = post_step1 - 2

            if bout_i < 0 or bout_i >= bout_nx or bout_j < 0 or bout_j >= bout_ny:
                continue

            p = matplotlib.patches.Polygon(
                np.concatenate((rm[i][j][idx], zm[i][j][idx])).reshape(2, 5).T,
                fill=True,
                closed=True,
            )
            patches.append(p)
            ii.append(bout_i)
            jj.append(bout_j)

    # ------------------------------------------------------------------
    # Branch-cut guard columns (close the top-of-domain wedge gap) — single
    # null only (bridge_cut).
    #
    # The two poloidal guard cells at the ny_inner branch cut
    # (gridue i = ny_inner+1, ny_inner+2) are skipped by the mapping above
    # because they carry no evolved data. In a single-null grid this cut sits at
    # the top of the poloidal domain where the plasma is poloidally continuous,
    # so dropping them leaves the visible wedge-shaped hole reported for LSN
    # plots. Bridge it: draw each guard column and colour it from its adjacent
    # interior column (inner guard from bout_j = ny_inner-1, outer guard from
    # bout_j = ny_inner). Mirrors the branch-cut handling in visualize_in_grid.py.
    # NOT done for double-null / snowflake, where ny_inner is a real target break.
    gi_in, gi_out = ny_inner + 1, ny_inner + 2
    bj_in, bj_out = ny_inner - 1, ny_inner
    if bridge_cut and gi_out < Nx_gridue and 0 <= bj_in < bout_ny and 0 <= bj_out < bout_ny:
        for j in range(1, Ny_gridue - 1):
            bout_i = j + 1
            if bout_i < 0 or bout_i >= bout_nx:
                continue
            for gi, bj in ((gi_in, bj_in), (gi_out, bj_out)):
                p = matplotlib.patches.Polygon(
                    np.concatenate((rm[gi][j][idx], zm[gi][j][idx])).reshape(2, 5).T,
                    fill=True,
                    closed=True,
                )
                patches.append(p)
                ii.append(bout_i)
                jj.append(bj)

    return patches, np.array(ii), np.array(jj)


# Per-cut colour convention (matches visualize_in_grid.py).
_BRANCH_CUT_COLORS = {
    "jyseps1_1": "gold",
    "jyseps2_1": "limegreen",
    "jyseps1_2": "cornflowerblue",
    "jyseps2_2": "tomato",
    "ny_inner": "cyan",
}


def plot_branch_cuts(ax, rm, zm, grid, ny_inner, lw=1.5, linestyle="--"):
    """Overlay the topological branch cuts on the cell edges (see sibling script)."""
    r = rm.transpose(1, 0, 2)
    z = zm.transpose(1, 0, 2)
    n_rad, n_pol = r.shape[0], r.shape[1]

    def gv(name):
        return int(grid[name].values) if name in grid else None

    def ipol(J):
        return J + 1 if J < ny_inner else J + 3

    ix1, ix2 = gv("ixseps1"), gv("ixseps2")
    cut_extent = {
        "jyseps1_1": ix1, "jyseps2_1": ix1,
        "jyseps1_2": ix2, "jyseps2_2": ix2,
    }

    drawn = False
    for name in ("jyseps1_1", "jyseps2_1", "jyseps1_2", "jyseps2_2"):
        J = gv(name)
        if J is None:
            continue
        ip = ipol(J)
        if not (0 <= ip < n_pol):
            continue
        ext = cut_extent[name]
        rad_max = n_rad if ext is None else min(ext, n_rad)
        ax.plot(r[0:rad_max, ip, 4], z[0:rad_max, ip, 4],
                color=_BRANCH_CUT_COLORS[name], lw=lw, ls=linestyle,
                label=f"{name}={J}", zorder=5)
        drawn = True

    if ny_inner is not None and 0 <= ny_inner + 1 < n_pol:
        ax.plot(r[:, ny_inner + 1, 3], z[:, ny_inner + 1, 3],
                color=_BRANCH_CUT_COLORS["ny_inner"], lw=lw, ls=linestyle,
                label=f"ny_inner={ny_inner}", zorder=5)
        drawn = True

    if drawn:
        ax.legend(loc="upper right", fontsize="small", framealpha=0.8)


def plot_separatrices(ax, rm, zm, grid, ny_inner, lw=1.5):
    """Overlay the separatrices (ixseps1 magenta, ixseps2 yellow) on the cell edges."""
    r = rm.transpose(1, 0, 2)
    z = zm.transpose(1, 0, 2)
    n_rad, n_pol = r.shape[0], r.shape[1]
    corner = 1

    def gv(name):
        return int(grid[name].values) if name in grid else None

    def ipol(J):
        return J + 1 if J < ny_inner else J + 3

    def plot_cols(k, cols, color, close=False, label=None):
        cols = [c for c in cols if 0 <= c < n_pol]
        if len(cols) < 2:
            return False
        R = [r[k, c, corner] for c in cols]
        Z = [z[k, c, corner] for c in cols]
        if close:
            R.append(R[0])
            Z.append(Z[0])
        ax.plot(R, Z, color=color, lw=lw, label=label, zorder=6)
        return True

    drawn = False

    ix1 = gv("ixseps1")
    if ix1 is not None and 0 <= ix1 - 1 < n_rad:
        k = ix1 - 1
        plot_cols(k, range(0, ny_inner + 2), "magenta", label=f"ixseps1={ix1}")
        plot_cols(k, range(ny_inner + 3, n_pol), "magenta")
        drawn = True

    ix2 = gv("ixseps2")
    if ix2 is not None and 0 <= ix2 - 1 < n_rad:
        k = ix2 - 1
        j11, j21 = gv("jyseps1_1"), gv("jyseps2_1")
        if j11 is not None and j21 is not None:
            plot_cols(k, range(ipol(j11 + 1), ipol(j21) + 1), "yellow",
                      close=True, label=f"ixseps2={ix2}")
            leg = list(range(0, ipol(j11) + 1)) + \
                list(range(ipol(j21 + 1), ny_inner + 2))
            plot_cols(k, leg, "yellow")
            plot_cols(k, range(ny_inner + 3, n_pol), "yellow")
        else:
            plot_cols(k, range(0, ny_inner + 1), "yellow", label=f"ixseps2={ix2}")
            plot_cols(k, range(ny_inner + 3, n_pol), "yellow")
        drawn = True

    if drawn:
        ax.legend(loc="upper right", fontsize="small", framealpha=0.8)


def _color_norm(values, log):
    """
    Build a colour ``Normalize`` for the convergence field.

    The timescale spans many orders of magnitude and is signed (the sign tells
    growing vs decaying), so when ``log`` is True we use a symmetric-log norm
    (:class:`matplotlib.colors.SymLogNorm`) centred on zero. Limits are taken
    from the 99th percentile of ``|values|`` so a handful of extreme cells (e.g.
    the near-converged ones that hit the ``1e-10`` denominator guard) don't wash
    out the rest, and ``linthresh`` (the half-width of the linear region around
    zero) is set from a low percentile of the non-zero magnitudes.

    Returns ``None`` for a plain linear scale (caller falls back to min/max).
    """
    v = values[np.isfinite(values)]
    if not log or v.size == 0:
        return None

    from matplotlib.colors import SymLogNorm

    absv = np.abs(v)
    nz = absv[absv > 0]
    if nz.size == 0:
        return None

    vmax = float(np.percentile(absv, 99))
    if vmax <= 0:
        return None
    linthresh = float(np.percentile(nz, 25))
    linthresh = min(linthresh, vmax)  # keep the linear region inside the range

    has_neg = bool((v < 0).any())
    vmin = -vmax if has_neg else float(max(nz.min(), vmax * 1e-6))
    return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)


def _decorate_axes(ax, rm, zm):
    """Common axis cosmetics shared by the static and animated plots."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(rm[:, :, 0].min(), rm[:, :, 0].max())
    ax.set_ylim(zm[:, :, 0].min(), zm[:, :, 0].max())


def polygon_plot(ax, rm, zm, data, ny_inner, cmap="viridis",
                 vmin=None, vmax=None, colorbar_label=None, log=False,
                 bridge_cut=False):
    """Create a 2D polygon plot from data of shape (bout_nx, bout_ny)."""
    bout_nx, bout_ny = data.shape
    patches, ii, jj = _build_patches(rm, zm, ny_inner, bout_nx, bout_ny,
                                     bridge_cut=bridge_cut)

    colors = data[ii, jj]
    norm = _color_norm(colors, log)

    collection = PatchCollection(
        patches, cmap=cmap, norm=norm, edgecolors="face",
        linewidths=0.1, joinstyle="bevel",
    )
    collection.set_array(colors)
    if norm is None:
        if vmin is None:
            vmin = np.nanmin(colors)
        if vmax is None:
            vmax = np.nanmax(colors)
        collection.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(collection)

    plt.colorbar(collection, ax=ax, label=colorbar_label, shrink=0.8)
    _decorate_axes(ax, rm, zm)

    return collection


def animate_plot(fig, ax, rm, zm, data_t, ny_inner, times=None,
                 cmap="viridis", colorbar_label=None, interval=200, log=False,
                 bridge_cut=False):
    """
    Animate the polygon plot over every available timestep. Patches and colour
    limits are fixed once (limits span the whole series); only per-cell colours
    update each frame.

    data_t has shape (nt, bout_nx, bout_ny).
    """
    from matplotlib.animation import FuncAnimation

    nt, bout_nx, bout_ny = data_t.shape
    patches, ii, jj = _build_patches(rm, zm, ny_inner, bout_nx, bout_ny,
                                     bridge_cut=bridge_cut)

    # Norm / colour limits fixed across the whole series so the colourbar is
    # stable from frame to frame.
    norm = _color_norm(data_t, log)

    collection = PatchCollection(
        patches, cmap=cmap, norm=norm, edgecolors="face",
        linewidths=0.1, joinstyle="bevel",
    )
    collection.set_array(data_t[0][ii, jj])
    if norm is None:
        collection.set_clim(vmin=np.nanmin(data_t), vmax=np.nanmax(data_t))
    ax.add_collection(collection)

    plt.colorbar(collection, ax=ax, label=colorbar_label, shrink=0.8)
    _decorate_axes(ax, rm, zm)
    title = ax.set_title("")

    def _frame_label(frame):
        if times is None:
            return f"frame {frame}/{nt - 1}"
        return f"t = {times[frame]:.4g}"

    def update(frame):
        collection.set_array(data_t[frame][ii, jj])
        title.set_text(f"{colorbar_label} ({_frame_label(frame)})")
        return collection, title

    anim = FuncAnimation(
        fig, update, frames=nt, interval=interval, blit=False, repeat=True,
    )
    return anim


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hermes-3 convergence-timescale polygon visualization: plots "
                    "var / (ddt(var) * Omega_ci) on the gridue mesh."
    )
    parser.add_argument("grid_file", type=str, help="Path to grid file (.nc)")
    parser.add_argument("sim_results", type=str,
                        help="Path to simulation results directory")
    parser.add_argument("--var", type=str, required=True, choices=ALLOWED_VARS,
                        help="Evolving variable to test. One of: "
                             + ", ".join(ALLOWED_VARS))
    parser.add_argument("--time", type=int, default=-1,
                        help="Time index (default: -1 = last common step)")
    parser.add_argument("--zindex", type=int, default=0,
                        help="Toroidal index (default: 0)")
    parser.add_argument("--cmap", type=str, default="Spectral_r",
                        help="Colormap (default: Spectral_r)")
    parser.add_argument("--animate", action="store_true", default=False,
                        help="Animate over all available (common) timesteps "
                             "instead of plotting a single one")
    parser.add_argument("--interval", type=int, default=200,
                        help="Delay between animation frames in ms "
                             "(only used with --animate; default: 200)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save the animation to this path (e.g. out.mp4 / "
                             "out.gif) instead of showing it (only with --animate)")
    parser.add_argument("--log", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use a symmetric-log colour scale so changes across "
                             "orders of magnitude are visible (default: on; "
                             "pass --no-log for a linear scale)")
    parser.add_argument("--plot_change", action="store_true", default=False,
                        help="Plot the fractional rate of change "
                             "(ddt(var)*Omega_ci)/var [1/s] instead of the "
                             "convergence timescale var/(ddt(var)*Omega_ci) [s].")
    parser.add_argument("--branch_cuts", action="store_true", default=False,
                        help="Overlay the topological branch cuts on the plot")
    parser.add_argument("--seps", action="store_true", default=False,
                        help="Overlay the separatrices (ixseps1, ixseps2) on the plot")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    # Load grid file and (time-aligned) simulation results.
    grid = xr.open_dataset(args.grid_file, engine="netcdf4")
    sim = open_aligned_boutdataset(args.sim_results)

    MXG = sim.metadata["MXG"]
    ny_inner = int(grid["ny_inner"].values)

    # Single-null signature: the two X-point poloidal indices coincide, so the
    # ny_inner branch cut is a continuous top-of-domain location that should be
    # bridged (see _build_patches). For a double null / snowflake it is a real
    # inner/outer target break and is left open.
    bridge_cut = ("jyseps2_1" in grid and "jyseps1_2" in grid
                  and int(grid["jyseps2_1"].values) == int(grid["jyseps1_2"].values))
    if bridge_cut:
        print("[Hermes-3-Conv-Test] Single-null grid (jyseps2_1 == jyseps1_2): "
              "bridging the ny_inner branch cut to close the top wedge gap.")

    rm = grid["rm"].values  # (Nx_gridue, Ny_gridue, 5)
    zm = grid["zm"].values

    omega_ci = get_omega_ci(sim)
    if args.plot_change:
        label = f"(ddt({args.var})·Omega_ci) / {args.var}  [1/s]"
    else:
        label = f"{args.var} / (ddt({args.var})·Omega_ci)  [s]"

    print(f"[Hermes-3-Conv-Test] Selected variable: --var {args.var}")
    print(f"[Hermes-3-Conv-Test] Omega_ci = {omega_ci:.6e} rad/s "
          f"(denominator guarded with +{_DENOM_EPS:g})")

    if args.animate:
        fig, ax = plt.subplots(figsize=(6, 10))
        data_t = convergence_field(sim, args.var, MXG, args.zindex,
                                   omega_ci, time=None,
                                   plot_change=args.plot_change)
        times = sim["t"].values if "t" in sim.coords else None

        anim = animate_plot(fig, ax, rm, zm, data_t, ny_inner, times=times,
                            cmap=args.cmap, colorbar_label=label,
                            interval=args.interval, log=args.log,
                            bridge_cut=bridge_cut)
        if args.branch_cuts:
            plot_branch_cuts(ax, rm, zm, grid, ny_inner)
        if args.seps:
            plot_separatrices(ax, rm, zm, grid, ny_inner)
        plt.tight_layout()
        if args.save:
            print(f"[Hermes-3-Conv-Test] Saving animation to {args.save} ...")
            anim.save(args.save)
            print("[Hermes-3-Conv-Test] Done.")
        else:
            plt.show()
    else:
        fig, ax = plt.subplots(figsize=(6, 10))
        data = convergence_field(sim, args.var, MXG, args.zindex,
                                 omega_ci, time=args.time,
                                 plot_change=args.plot_change)
        polygon_plot(ax, rm, zm, data, ny_inner,
                     cmap=args.cmap, colorbar_label=label, log=args.log,
                     bridge_cut=bridge_cut)
        if args.branch_cuts:
            plot_branch_cuts(ax, rm, zm, grid, ny_inner)
        if args.seps:
            plot_separatrices(ax, rm, zm, grid, ny_inner)
        kind = "rate" if args.plot_change else "convergence"
        ax.set_title(f"{args.var} {kind} (t={args.time}, z={args.zindex})")
        plt.tight_layout()
        plt.show()
