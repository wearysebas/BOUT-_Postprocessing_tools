#!/usr/bin/env python3
"""
BOUT++ 2D animated polygon visualization for INGRID Single Null grids.

Builds polygon patches once from the INGRID gridue rm/zm arrays, then animates
by updating the color array at each time step.

Uses the same direct BOUT↔gridue index mapping as BOUT_postprocessing_in_SN.py:
  j_rad  = ix - 1
  i_pol  = iy + 1

Usage:
    python BOUT_in_animation.py <grid_file> <sim_results_dir> [options]

Example:
    python BOUT_in_animation.py \\
        SN/INGRID/IN_try3_bout_from_in.grd.nc SN/INGRID \\
        --var T --zindex 0 --interval 100
"""

import numpy as np
import xarray as xr
import xbout
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection


def build_patches_and_colors(rm, zm, data_2d, ny_inner, MXG):
    """
    Build polygon patches and extract colors from a 2D data slice.

    Uses the same logic as polygon_plot_sn in BOUT_postprocessing_in_SN.py:
    iterates over data.shape (not grid shape) and builds patches
    and colors together in lock-step.

    Parameters
    ----------
    rm, zm : ndarray, shape (n_pol, n_rad, 5)
        INGRID gridue arrays.  Last-axis index 0 = cell centre, 1-4 = corners.
    data_2d : np.ndarray, shape (nx, ny)
        Simulation data at a single time and z index.
    ny_inner : int
        Number of poloidal cells in the inner half (from the grid file).
    MXG : int
        Number of radial guard cells on each side to skip.

    Returns
    -------
    patches : list of matplotlib.patches.Polygon
    colors : np.ndarray
    indices : list of (int, int)
    R_centers : list of float
    Z_centers : list of float
    """
    nx, ny = data_2d.shape
    k = [1, 2, 4, 3]  # corner order: NW→SW→SE→NE (same as BOUT_postprocessing_in_SN.py)

    patches = []
    colors = []
    indices = []
    R_centers = []
    Z_centers = []

    for ix in range(MXG, nx - MXG):
        j_rad = ix - 1
        for iy in range(ny):
            i_pol = iy + 1
            verts = np.column_stack([rm[i_pol, j_rad, k], zm[i_pol, j_rad, k]])
            patches.append(matplotlib.patches.Polygon(verts, closed=True))
            colors.append(data_2d[ix, iy])
            indices.append((ix, iy))
            R_centers.append(rm[i_pol, j_rad, 0])
            Z_centers.append(zm[i_pol, j_rad, 0])

    return patches, np.array(colors), indices, R_centers, Z_centers


def extract_colors(data_2d, indices):
    """Extract color values from a 2D data slice for each polygon."""
    return np.array([data_2d[ix, iy] for ix, iy in indices])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BOUT++ 2D animated polygon visualization for INGRID Single Null grids"
    )
    parser.add_argument("grid_file", type=str,
                        help="Path to INGRID grid file (.nc)")
    parser.add_argument("sim_results", type=str,
                        help="Path to simulation results directory")
    parser.add_argument("--var", type=str, default="T",
                        help="Variable to plot (default: T)")
    parser.add_argument("--zindex", type=int, default=0,
                        help="Toroidal index (default: 0)")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Colormap (default: viridis)")
    parser.add_argument("--interval", type=int, default=100,
                        help="Delay between frames in ms (default: 100)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save animation to file (e.g. output.mp4 or output.gif)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for saved animation (default: 10)")
    parser.add_argument("--update_colorbar", type=bool, default=False,
                        help="Update colorbar limits for each frame")

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

    rm = grid["rm"].values  # (n_pol, n_rad, 5)
    zm = grid["zm"].values

    # All time steps: shape (nt, nx, ny) — x includes guard cells
    all_data = sim[args.var].values[:, :, :, args.zindex]
    nt = all_data.shape[0]
    times = sim["t"].values

    # Build patches and colors from time 0 (same logic as polygon_plot_sn)
    patches, colors0, indices, R_centers, Z_centers = build_patches_and_colors(
        rm, zm, all_data[0], ny_inner, MXG
    )

    # Fix color scale from time 0
    vmin = np.nanmin(colors0)
    vmax = np.nanmax(colors0)

    # Set up figure — identical to polygon_plot_sn
    fig, ax = plt.subplots(figsize=(6, 10))

    collection = PatchCollection(
        patches, cmap=args.cmap, edgecolors="face",
        linewidths=0.1, joinstyle="bevel",
    )
    collection.set_array(colors0)
    collection.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(collection)

    plt.colorbar(collection, ax=ax, label=args.var, shrink=0.8)

    # Branch-cut guard cells: unfilled outlines to close the visual gap
    nx_data, ny_data = all_data[0].shape
    k = [1, 2, 4, 3]
    gc_patches = []
    for ix in range(MXG, nx_data - MXG):
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

    title = ax.set_title(f"{args.var}  t = {times[0]:.3f}")
    plt.tight_layout()

    def update(frame, update_colorbar=False):
        colors = extract_colors(all_data[frame], indices)
        collection.set_array(colors)
        if update_colorbar == True:
            collection.set_clim(vmin=vmin, vmax=np.nanmax(colors))
        title.set_text(f"{args.var}  t = {times[frame]:.3f}")
        return collection, title

    anim = animation.FuncAnimation(
        fig, update, frames=nt, interval=args.interval, blit=False, fargs=(args.update_colorbar,)
    )

    if args.save:
        print(f"Saving animation to {args.save} ({nt} frames)...")
        if args.save.endswith(".gif"):
            anim.save(args.save, writer="pillow", fps=args.fps)
        else:
            anim.save(args.save, writer="ffmpeg", fps=args.fps)
        print("Done.")
    else:
        plt.show()
