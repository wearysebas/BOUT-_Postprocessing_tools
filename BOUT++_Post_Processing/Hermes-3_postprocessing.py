#!/usr/bin/env python3
"""
Hermes-3 2D polygon visualization using grid cell corners from gridue data (rm, zm).

This is a Hermes-3 / snowflake-geometry-aware variant of
``BOUT_postprocessing_sdtools.py``. The polygon plotting is identical; the only
difference is how the simulation output is loaded.

Why a separate loader is needed
--------------------------------
When a Hermes-3 run is killed (or wraps up) on an HPC, the individual MPI ranks
do not always flush the same number of output steps to their ``BOUT.dmp.<n>.nc``
files. xbout combines the per-processor files *spatially* (along x and y) and
assumes the time dimension is identical across every file, so a mismatch in the
length of ``t`` makes ``open_boutdataset`` raise (the "exact"-join alignment
fails). This is more likely with the PETSc solver, where the last reported
output need not land on every rank simultaneously.

The fix: open each dump file individually, find the largest time index that is
present in *every* file (i.e. ``min`` of the per-file ``t`` lengths), truncate
all of them to that common length, and hand the already-aligned list of
datasets to xbout. xbout's ``open_boutdataset`` accepts a list of in-memory
``xr.Dataset`` objects and runs the exact same guard-cell trimming / nested
combination it would for a glob, so the resulting dataset is identical to the
normal path apart from being clipped to the last common timestep.

Usage:
    python Hermes-3_postprocessing.py <grid_file> <sim_results_dir> [options]

Example:
    python Hermes-3_postprocessing.py \
        MAST-U_SF75.grd.nc Bender_mesh_Hermes/examples/... \
        --var T --time -1 --zindex 0
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


def _proc_index(path):
    """Sort key: numeric processor index from a BOUT.dmp.<n>.nc filename."""
    m = re.search(r"BOUT\.dmp\.(\d+)\.nc$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def open_aligned_boutdataset(datapath, **kwargs):
    """
    Load a Hermes-3 / BOUT++ run whose per-processor dump files may contain
    different numbers of time steps.

    Parameters
    ----------
    datapath : str
        Either a directory containing ``BOUT.dmp.*.nc`` files, or a glob
        pattern (e.g. ``".../BOUT.dmp.*.nc"``).
    **kwargs
        Forwarded verbatim to :func:`xbout.open_boutdataset` (e.g.
        ``keep_xboundaries``, ``keep_yboundaries``, ``info``).

    Returns
    -------
    xarray.Dataset
        The combined dataset, truncated to the last time index that is present
        in every dump file.
    """
    # Accept a directory, a glob pattern, or a path to a single representative file.
    if os.path.isdir(datapath):
        pattern = os.path.join(datapath, "BOUT.dmp.*.nc")
    else:
        pattern = datapath

    files = sorted(glob.glob(pattern), key=_proc_index)
    if not files:
        raise FileNotFoundError(f"No BOUT.dmp files matched: {pattern!r}")

    # Open each per-processor file separately so we can inspect/clip its time axis.
    # netCDF data variables are read lazily, so this does not pull everything
    # into memory.
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
            f"[Hermes-3_postprocessing] Time-step mismatch across dump files: "
            f"lengths range {min_t}..{max_t} "
            f"({n_short}/{len(tlens)} ranks short of the maximum). "
            f"Truncating all files to the last common step (t index 0..{min_t - 1})."
        )
    else:
        print(
            f"[Hermes-3_postprocessing] All {len(files)} dump files agree on "
            f"{min_t} time steps; no truncation needed."
        )

    # Clip every file to the common time range, then let xbout do the spatial
    # combination + guard-cell handling exactly as it would for a glob.
    datasets = [ds.isel(t=slice(0, min_t)) for ds in datasets]

    return xbout.open_boutdataset(datapath=datasets, **kwargs)


def _build_patches(rm, zm, ny_inner, bout_nx, bout_ny):
    """
    Build the polygon patches and their (bout_i, bout_j) data indices by
    iterating over the raw gridue cells.

    The geometry does not change with time, so this is computed once and reused
    for every frame when animating.

    Gridue rm/zm have shape (Nx_gridue, Ny_gridue, 5) where:
      - First dim is poloidal (BOUT y direction)
      - Second dim is radial (BOUT x direction)
      - Third dim: 0=center, 1-4=corners

    Corner layout:
        (1) -- (3)
         |      |
         |  (0) |  -> Radial (BOUT x)
         |      |
        (2) -- (4)

    Mapping from gridue (i, j) to BOUT (bout_i, bout_j):
      Radial:   gridue j=1..Ny-2  -> bout_i = j + 1
      Poloidal: gridue i=1..Nx-2  -> bout_j depends on ny_inner:
                  post_step1 = i - 1
                  if post_step1 < ny_inner:        bout_j = post_step1
                  if post_step1 == ny_inner or ny_inner+1: guard cell (skip)
                  if post_step1 > ny_inner+1:     bout_j = post_step1 - 2

    Returns
    -------
    patches : list[matplotlib.patches.Polygon]
    ii, jj : np.ndarray
        Index arrays such that ``data[ii, jj]`` gives the colour for each patch,
        for ``data`` of shape ``(bout_nx, bout_ny)``.
    """
    Nx_gridue = rm.shape[0]
    Ny_gridue = rm.shape[1]

    patches = []
    ii = []
    jj = []
    idx = [np.array([1, 2, 4, 3, 1])]

    for i in range(Nx_gridue):
        for j in range(Ny_gridue):
            # Map gridue (i, j) -> BOUT (bout_i, bout_j)

            # Radial: skip boundary cells (j=0 and j=Ny-1)
            if j == 0 or j == Ny_gridue - 1:
                continue
            bout_i = j + 1  # gridue j=1 -> bout_i=2, ..., j=Ny-2 -> bout_i=Ny-1

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

            # Bounds check
            if bout_i < 0 or bout_i >= bout_nx or bout_j < 0 or bout_j >= bout_ny:
                continue

            # Build polygon from actual gridue corners
            p = matplotlib.patches.Polygon(
                np.concatenate((rm[i][j][idx], zm[i][j][idx])).reshape(2, 5).T,
                fill=True,
                closed=True,
            )
            patches.append(p)
            ii.append(bout_i)
            jj.append(bout_j)

    return patches, np.array(ii), np.array(jj)


# Per-cut colour convention (matches visualize_in_grid.py). Each branch cut gets
# its own colour; all drawn dashed.
_BRANCH_CUT_COLORS = {
    "jyseps1_1": "gold",
    "jyseps2_1": "limegreen",
    "jyseps1_2": "cornflowerblue",
    "jyseps2_2": "tomato",
    "ny_inner": "cyan",
}


def plot_branch_cuts(ax, rm, zm, grid, ny_inner, lw=1.5, linestyle="--"):
    """
    Overlay the topological branch cuts on an existing polygon plot, drawing
    each on the *cell edge* (not the cell centre).

    This follows the edge-drawing convention of ``visualize_grid_Modified.py``:
    the gridue corner arrays are transposed to ``[radial, poloidal, corner]``
    and the cut is drawn along a fixed poloidal column. Corner index 4 (the
    radial-running cell face) is used for the ``jyseps`` cuts; corner index 3
    (the poloidal-facing / target face) for the ``ny_inner`` inner-target line.

    The poloidal column for a BOUT y-cut at index ``J`` accounts for the gridue
    guard cells: ``J + 1`` below ``ny_inner``, ``J + 3`` above it.

    Radial extent (SF topology, taken from the grid rather than hard-coded):
      - ``jyseps1_1`` / ``jyseps2_1`` run from the inner boundary out to
        ``ixseps1`` (the primary / SOL separatrix);
      - ``jyseps1_2`` / ``jyseps2_2`` run out to ``ixseps2``;
      - ``ny_inner`` (the inner/outer target split) spans the full radial width
        (0 → nx), drawn on the target face.
    """
    # Transpose to [radial, poloidal, corner] so a fixed poloidal column gives a
    # radial line, exactly as in visualize_grid_Modified.py.
    r = rm.transpose(1, 0, 2)
    z = zm.transpose(1, 0, 2)
    n_rad, n_pol = r.shape[0], r.shape[1]

    def gv(name):
        return int(grid[name].values) if name in grid else None

    # gridue poloidal column for a BOUT y-cut at index J (guard-cell offset).
    def ipol(J):
        return J + 1 if J < ny_inner else J + 3

    # Radial extent (separatrix index) each cut terminates at.
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

    # ny_inner: inner/outer divertor target split, full radial extent, target face.
    if ny_inner is not None and 0 <= ny_inner + 1 < n_pol:
        ax.plot(r[:, ny_inner + 1, 3], z[:, ny_inner + 1, 3],
                color=_BRANCH_CUT_COLORS["ny_inner"], lw=lw, ls=linestyle,
                label=f"ny_inner={ny_inner}", zorder=5)
        drawn = True

    if drawn:
        ax.legend(loc="upper right", fontsize="small", framealpha=0.8)


def plot_separatrices(ax, rm, zm, grid, ny_inner, lw=1.5):
    """
    Overlay the separatrices (``ixseps1`` magenta, ``ixseps2`` yellow) on the
    plot, on the cell edge (corner 1 of the cells at radial index ``ixseps-1``
    in the transposed corner array).

    ``ixseps1`` is the open SOL separatrix: it spans the whole poloidal domain,
    drawn in two arcs split at the ``ny_inner`` inner/outer target gap.

    ``ixseps2`` bounds the closed core, so it is drawn in pieces around the
    lower X-point, which reconnects the inner and outer sides differently:
      - core ring: the arc ``jyseps1_1+1 .. jyseps2_1`` closed on itself (ends
        where it starts, at ``jyseps2_1``);
      - leg: the lower leg continues across the X-point on the outer side, i.e.
        ``jyseps1_1`` is joined to ``jyseps2_1+1``.
    The outer (above ``ny_inner``) half is drawn continuously.
    """
    r = rm.transpose(1, 0, 2)
    z = zm.transpose(1, 0, 2)
    n_rad, n_pol = r.shape[0], r.shape[1]
    corner = 1  # separatrix edge

    def gv(name):
        return int(grid[name].values) if name in grid else None

    def ipol(J):
        return J + 1 if J < ny_inner else J + 3

    def plot_cols(k, cols, color, close=False, label=None):
        """Plot the separatrix (radial index k) along a list of poloidal columns."""
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

    # -- ixseps1: open SOL separatrix, inner + outer arcs (split at ny_inner gap) --
    ix1 = gv("ixseps1")
    if ix1 is not None and 0 <= ix1 - 1 < n_rad:
        k = ix1 - 1
        plot_cols(k, range(0, ny_inner + 2), "magenta", label=f"ixseps1={ix1}")
        plot_cols(k, range(ny_inner + 3, n_pol), "magenta")
        drawn = True

    # -- ixseps2: closed core ring + leg joined across the lower X-point --
    ix2 = gv("ixseps2")
    if ix2 is not None and 0 <= ix2 - 1 < n_rad:
        k = ix2 - 1
        j11, j21 = gv("jyseps1_1"), gv("jyseps2_1")
        if j11 is not None and j21 is not None:
            # Core ring: closed arc between the X-points.
            plot_cols(k, range(ipol(j11 + 1), ipol(j21) + 1), "yellow",
                      close=True, label=f"ixseps2={ix2}")
            # Lower leg: below jyseps1_1, continuing from jyseps2_1+1 (the jump
            # between the two ranges draws the jyseps1_1 -> jyseps2_1+1 join).
            leg = list(range(0, ipol(j11) + 1)) + \
                list(range(ipol(j21 + 1), ny_inner + 2))
            plot_cols(k, leg, "yellow")
            # Outer half (above ny_inner), drawn continuously.
            plot_cols(k, range(ny_inner + 3, n_pol), "yellow")
        else:
            plot_cols(k, range(0, ny_inner + 1), "yellow", label=f"ixseps2={ix2}")
            plot_cols(k, range(ny_inner + 3, n_pol), "yellow")
        drawn = True

    if drawn:
        ax.legend(loc="upper right", fontsize="small", framealpha=0.8)


def _convergence_line(da, name, name_w=0):
    """
    Format one variable's first-vs-last-timestep comparison as::

        <name>: <first> - <last> : <first - last>

    ``<first>``/``<last>`` are the values at ``t=0`` and ``t=-1``. Fields with
    spatial extent are reduced to a single representative number with
    ``nanmean`` (flagged with ``<mean>``); genuine scalars are printed as-is.
    ``name_w`` pads the name so multiple lines line up in a column.
    """
    first = da.isel(t=0)
    last = da.isel(t=-1)
    if da.ndim > 1:
        first_v = float(np.nanmean(first.values))
        last_v = float(np.nanmean(last.values))
        tag = " <mean>"
    else:
        first_v = float(first.values)
        last_v = float(last.values)
        tag = ""
    diff = first_v - last_v
    return f"  {name:<{name_w}}{tag}: {first_v: .6e} - {last_v: .6e} : {diff: .6e}"


def print_convergence(sim):
    """
    Print, for every time-dependent variable in the dump files, how much it
    changed between the first and the last (common) timestep.
    """
    # Only variables that actually carry a time axis can be compared.
    tvars = sorted(name for name, da in sim.data_vars.items() if "t" in da.dims)
    if not tvars:
        print("[Hermes-3_postprocessing] No time-dependent variables to compare.")
        return

    name_w = max(len(n) for n in tvars)
    print(
        f"[Hermes-3_postprocessing] Convergence check "
        f"(first - last : difference), '<mean>' = spatially averaged field:"
    )
    for name in tvars:
        print(_convergence_line(sim[name], name, name_w))


def _decorate_axes(ax, rm, zm):
    """Common axis cosmetics shared by the static and animated plots."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(rm[:, :, 0].min(), rm[:, :, 0].max())
    ax.set_ylim(zm[:, :, 0].min(), zm[:, :, 0].max())


def polygon_plot(ax, rm, zm, data, ny_inner, cmap="viridis",
                 vmin=None, vmax=None, colorbar_label=None):
    """
    Create a 2D polygon plot by iterating over raw gridue cells.

    data has shape (bout_nx, bout_ny) = (nx, ny) after stripping MXG guard cells.
    """
    bout_nx, bout_ny = data.shape
    patches, ii, jj = _build_patches(rm, zm, ny_inner, bout_nx, bout_ny)

    colors = data[ii, jj]
    if vmin is None:
        vmin = np.nanmin(colors)
    if vmax is None:
        vmax = np.nanmax(colors)

    collection = PatchCollection(
        patches, cmap=cmap, edgecolors="face",
        linewidths=0.1, joinstyle="bevel",
    )
    collection.set_array(colors)
    collection.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(collection)

    plt.colorbar(collection, ax=ax, label=colorbar_label, shrink=0.8)

    _decorate_axes(ax, rm, zm)

    return collection


def animate_plot(fig, ax, rm, zm, data_t, ny_inner, times=None,
                 cmap="viridis", colorbar_label=None, interval=200):
    """
    Animate the polygon plot over every available timestep.

    The patches and colour limits are fixed once (limits span the whole time
    series so the colourbar is stable across frames); only the per-cell colours
    are updated each frame, which keeps the animation cheap.

    Parameters
    ----------
    data_t : np.ndarray
        Shape ``(nt, bout_nx, bout_ny)`` — the variable over all timesteps,
        already stripped of x guard cells and sliced in z.
    times : array-like or None
        Physical time values for the title; if None, the frame index is shown.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Keep a reference to this alive while the figure is shown/saved.
    """
    from matplotlib.animation import FuncAnimation

    nt, bout_nx, bout_ny = data_t.shape
    patches, ii, jj = _build_patches(rm, zm, ny_inner, bout_nx, bout_ny)

    # Fixed colour limits across the whole time series.
    vmin = np.nanmin(data_t)
    vmax = np.nanmax(data_t)

    collection = PatchCollection(
        patches, cmap=cmap, edgecolors="face",
        linewidths=0.1, joinstyle="bevel",
    )
    collection.set_array(data_t[0][ii, jj])
    collection.set_clim(vmin=vmin, vmax=vmax)
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
        title.set_text(f"{colorbar_label} ({_frame_label(frame)}, z fixed)")
        return collection, title

    anim = FuncAnimation(
        fig, update, frames=nt, interval=interval, blit=False, repeat=True,
    )
    return anim


def _str2bool(value):
    """Parse a CLI boolean so ``--plot False`` / ``--test_convergence True`` work."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hermes-3 2D polygon visualization using gridue cell corners "
                    "(handles per-rank time-step mismatch in BOUT.dmp files)"
    )
    parser.add_argument("grid_file", type=str, help="Path to grid file (.nc)")
    parser.add_argument("sim_results", type=str,
                        help="Path to simulation results directory")
    parser.add_argument("--var", type=str, default=None,
                        help="Variable to plot (default: T). With "
                             "--test_convergence, specifying --var reports only "
                             "that variable; omitting it reports the whole list.")
    parser.add_argument("--time", type=int, default=-1,
                        help="Time index (default: -1 = last common step)")
    parser.add_argument("--zindex", type=int, default=0,
                        help="Toroidal index (default: 0)")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Colormap (default: viridis)")
    parser.add_argument("--animate", action="store_true", default=False,
                        help="Animate over all available (common) timesteps "
                             "instead of plotting a single one")
    parser.add_argument("--interval", type=int, default=200,
                        help="Delay between animation frames in ms "
                             "(only used with --animate; default: 200)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save the animation to this path (e.g. out.mp4 / "
                             "out.gif) instead of showing it (only with --animate)")
    parser.add_argument("--branch_cuts", action="store_true", default=False,
                        help="Overlay the topological branch cuts "
                             "(jyseps1_1/2_1/1_2/2_2 + ny_inner) on the plot")
    parser.add_argument("--seps", action="store_true", default=False,
                        help="Overlay the separatrices (ixseps1, ixseps2) on the plot")
    parser.add_argument("--plot", type=_str2bool, default=True,
                        metavar="True/False",
                        help="Show the plot (default: True); pass --plot False to "
                             "skip all plotting (e.g. when only --test_convergence "
                             "is wanted)")
    parser.add_argument("--test_convergence", type=_str2bool, default=False,
                        metavar="True/False",
                        help="Print, for every time-dependent variable, its first "
                             "vs last timestep value and their difference "
                             "(default: False)")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    # Load grid file and simulation results.
    # The simulation is loaded via the time-aligning loader so that dump files
    # which ended on different steps don't crash the combine.
    grid = xr.open_dataset(args.grid_file, engine="netcdf4")
    sim = open_aligned_boutdataset(args.sim_results)

    # A variable was explicitly requested only if --var was passed; otherwise it
    # falls back to "T" for plotting but means "no variable specified" for the
    # convergence report.
    var_specified = args.var is not None
    if args.var is None:
        args.var = "T"

    MXG = sim.metadata["MXG"]
    nx = int(grid["nx"].values)
    ny = int(grid["ny"].values)
    ny_inner = int(grid["ny_inner"].values)

    # Raw gridue cell corners from grid file
    rm = grid["rm"].values  # (Nx_gridue, Ny_gridue, 5)
    zm = grid["zm"].values

    print(f"[Hermes-3_postprocessing] Selected variable: --var {args.var}")
    # Convergence report: if a variable was explicitly given, report only that
    # one (regardless of whether the plot is shown); otherwise dump the whole
    # list.
    if args.test_convergence:
        if not var_specified:
            print_convergence(sim)
        elif args.var in sim.data_vars and "t" in sim[args.var].dims:
            print(f"[Hermes-3_postprocessing] Convergence of '{args.var}':")
            print(_convergence_line(sim[args.var], args.var))
        else:
            print(
                f"[Hermes-3_postprocessing] '{args.var}' has no time axis; "
                f"nothing to compare."
            )

    if not args.plot:
        # --plot False: skip all figure work (e.g. a headless --test_convergence run).
        print("[Hermes-3_postprocessing] --plot False: skipping plotting.")
    elif args.animate:
        fig, ax = plt.subplots(figsize=(6, 10))
        # All timesteps: strip x guard cells, slice in z -> (nt, bout_nx, bout_ny).
        # The loader already clipped to the last step common to every dump file.
        data_t = sim[args.var].values[:, MXG:-MXG, :, args.zindex]
        times = sim["t"].values if "t" in sim.coords else None

        anim = animate_plot(fig, ax, rm, zm, data_t, ny_inner, times=times,
                            cmap=args.cmap, colorbar_label=args.var,
                            interval=args.interval)
        if args.branch_cuts:
            plot_branch_cuts(ax, rm, zm, grid, ny_inner)
        if args.seps:
            plot_separatrices(ax, rm, zm, grid, ny_inner)
        plt.tight_layout()
        if args.save:
            print(f"[Hermes-3_postprocessing] Saving animation to {args.save} ...")
            anim.save(args.save)
            print("[Hermes-3_postprocessing] Done.")
        else:
            plt.show()
    else:
        fig, ax = plt.subplots(figsize=(6, 10))
        # Single timestep: strip x guard cells, select time and z slice
        data = sim[args.var].values[args.time, MXG:-MXG, :, args.zindex]
        polygon_plot(ax, rm, zm, data, ny_inner,
                     cmap=args.cmap, colorbar_label=args.var)
        if args.branch_cuts:
            plot_branch_cuts(ax, rm, zm, grid, ny_inner)
        if args.seps:
            plot_separatrices(ax, rm, zm, grid, ny_inner)
        ax.set_title(f"{args.var} (t={args.time}, z={args.zindex})")
        plt.tight_layout()
        plt.show()
