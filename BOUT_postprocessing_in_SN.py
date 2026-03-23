#!/usr/bin/env python3
"""
BOUT++ 2D polygon visualization for INGRID Single Null grids.

Uses the same direct BOUT↔gridue index mapping as visualize_in_grid.py:
  j_rad  = ix - 1
  i_pol  = iy + 1   (if iy < ny_inner)
  i_pol  = iy + 3   (if iy >= ny_inner)

The two branch-cut guard cells at i_pol = ny_inner+1 and ny_inner+2 are
plotted as unfilled outlines to avoid a visible gap at the branch cut,
matching the behaviour of visualize_in_grid.py.

Usage:
    python BOUT_postprocessing_in_SN.py <grid_file> <sim_results_dir> [options]

Example:
    python BOUT_postprocessing_in_SN.py \\
        SN/INGRID/IN_try3_bout_from_in.grd.nc SN/INGRID \\
        --var T --time -1 --zindex 0
"""

import numpy as np
import xarray as xr
import xbout
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


def polygon_plot_sn(ax, rm, zm, data, ny_inner, MXG,
                    cmap="viridis", vmin=None, vmax=None, colorbar_label=None):
    """
    Create a 2D polygon plot for an INGRID Single Null gridue.

    Uses the direct BOUT↔gridue index mapping from visualize_in_grid.py:
      j_rad  = ix - 1
      i_pol  = iy + 1   if iy < ny_inner
      i_pol  = iy + 3   if iy >= ny_inner

    The two branch-cut guard cells (i_pol = ny_inner+1, ny_inner+2) have no
    simulation data but are drawn as unfilled outlines so the grid appears
    visually continuous across the branch cut.

    Parameters
    ----------
    ax : matplotlib Axes
    rm, zm : ndarray, shape (n_pol, n_rad, 5)
        INGRID gridue arrays.  Last-axis index 0 = cell centre, 1-4 = corners.
    data : ndarray, shape (nx, ny)
        Simulation data with full x dimension (including MXG guard cells).
    ny_inner : int
        Number of poloidal cells in the inner half (from the grid file).
    MXG : int
        Number of radial x guard cells on each side (typically 2).
    """
    nx, ny = data.shape
    k = [1, 2, 4, 3]   # corner order: NW→SW→SE→NE  (same as visualize_in_grid.py)

    patches = []
    colors = []
    R_centers = []
    Z_centers = []

    for ix in range(MXG, nx - MXG):
        j_rad = ix - 1
        for iy in range(ny):
            i_pol = (iy + 1)
            verts = np.column_stack([rm[i_pol, j_rad, k], zm[i_pol, j_rad, k]])
            patches.append(matplotlib.patches.Polygon(verts, closed=True))
            colors.append(data[ix, iy])
            R_centers.append(rm[i_pol, j_rad, 0])
            Z_centers.append(zm[i_pol, j_rad, 0])

    colors = np.array(colors)
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

    # Branch-cut guard cells: unfilled outlines to close the visual gap,
    # matching the approach in visualize_in_grid.py / build_branch_cut_polygons.
    gc_patches = []
    for ix in range(MXG, nx - MXG):
        j_rad = ix - 1
        for i_pol_gc in [ny_inner + 1, ny_inner + 2]:
            verts = np.column_stack([rm[i_pol_gc, j_rad, k], zm[i_pol_gc, j_rad, k]])
            gc_patches.append(matplotlib.patches.Polygon(verts, closed=True))
    gc_col = PatchCollection(gc_patches, facecolor="none",
                             edgecolors="face", linewidths=0.1)
    ax.add_collection(gc_col)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(min(R_centers), max(R_centers))
    ax.set_ylim(min(Z_centers), max(Z_centers))

    return collection


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BOUT++ 2D polygon visualization for INGRID Single Null grids"
    )
    parser.add_argument("grid_file", type=str, help="Path to grid file (.nc)")
    parser.add_argument("sim_results", type=str,
                        help="Path to simulation results directory")
    parser.add_argument("--var", type=str, default="T",
                        help="Variable to plot (default: T)")
    parser.add_argument("--time", type=int, default=-1,
                        help="Time index (default: -1 = last)")
    parser.add_argument("--zindex", type=int, default=0,
                        help="Toroidal index (default: 0)")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Colormap (default: viridis)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Color scale minimum (default: auto)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Color scale maximum (default: auto)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to this path instead of showing")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    grid = xr.open_dataset(args.grid_file, engine="netcdf4")
    sim = xbout.open_boutdataset(args.sim_results + "/BOUT.dmp.*.nc")

    MXG = sim.metadata["MXG"]
    ny_inner = int(grid["ny_inner"].values)

    # Keep full x dimension; guard-cell exclusion is handled by the index mapping
    data = sim[args.var].values[args.time, :, :, args.zindex]

    rm = grid["rm"].values    # (n_pol, n_rad, 5)
    zm = grid["zm"].values

    fig, ax = plt.subplots(figsize=(6, 10))
    polygon_plot_sn(
        ax, rm, zm, data, ny_inner, MXG,
        cmap=args.cmap, colorbar_label=args.var,
        vmin=args.vmin, vmax=args.vmax,
    )
    ax.set_title(f"{args.var} (t={args.time}, z={args.zindex})")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()
