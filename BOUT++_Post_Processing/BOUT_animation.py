#!/usr/bin/env python3
"""
BOUT++ 2D animated polygon visualization using grid cell corners from gridue data.

Builds polygon patches once from the raw gridue cell corners, then animates
by updating the color array at each time step.

Usage:
    python BOUT_animation.py <grid_file> <sim_results_dir> [options]

Example:
    python BOUT_animation.py \
        1st_try/gridue_try6_bout_from_in.grd.nc 1st_try \
        --var T --zindex 0 --interval 100
"""

import numpy as np
import xarray as xr
import xbout
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection


def build_polygons(rm, zm, ny_inner, bout_nx, bout_ny):
    """
    Build polygon patches from raw gridue cells and return the index mapping.

    Returns:
        patches: list of matplotlib Polygon objects (geometry only)
        bout_indices: list of (bout_i, bout_j) tuples matching each patch
    """
    Nx_gridue = rm.shape[0]
    Ny_gridue = rm.shape[1]

    patches = []
    bout_indices = []
    idx = [np.array([1, 2, 4, 3, 1])]

    for i in range(Nx_gridue):
        for j in range(Ny_gridue):
            # Radial: skip boundary cells
            if j == 0 or j == Ny_gridue - 1:
                continue
            bout_i = j + 1

            # Poloidal: skip boundary and guard cells
            if i == 0 or i == Nx_gridue - 1:
                continue
            post_step1 = i - 1
            if post_step1 < ny_inner:
                bout_j = post_step1
            elif post_step1 == ny_inner or post_step1 == ny_inner + 1:
                continue
            else:
                bout_j = post_step1 - 2

            if bout_i < 0 or bout_i >= bout_nx or bout_j < 0 or bout_j >= bout_ny:
                continue

            p = matplotlib.patches.Polygon(
                np.concatenate((rm[i][j][idx], zm[i][j][idx])).reshape(2, 5).T,
                fill=True, closed=True,
            )
            patches.append(p)
            bout_indices.append((bout_i, bout_j))

    return patches, bout_indices


def extract_colors(data_2d, bout_indices):
    """Extract color values from a 2D data slice for each polygon."""
    return np.array([data_2d[bi, bj] for bi, bj in bout_indices])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BOUT++ 2D animated polygon visualization"
    )
    parser.add_argument("grid_file", type=str, help="Path to grid file (.nc)")
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

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    # Load grid file and simulation results
    grid = xr.open_dataset(args.grid_file, engine="netcdf4")
    sim = xbout.open_boutdataset(args.sim_results + "/BOUT.dmp.*.nc")

    MXG = sim.metadata["MXG"]
    nx = int(grid["nx"].values)
    ny = int(grid["ny"].values)
    ny_inner = int(grid["ny_inner"].values)

    # Load all time steps: shape (nt, bout_nx, bout_ny)
    all_data = sim[args.var].values[:, MXG:-MXG, :, args.zindex]
    nt = all_data.shape[0]
    times = sim["t"].values

    # Raw gridue cell corners
    rm = grid["rm"].values
    zm = grid["zm"].values

    # Build polygons once
    patches, bout_indices = build_polygons(rm, zm, ny_inner, nx, ny)

    # Global minimum (fixed), per-frame maximum (dynamic)
    vmin = np.nanmin(all_data)

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 10))

    collection = PatchCollection(
        patches, cmap=args.cmap, edgecolors="face",
        linewidths=0.1, joinstyle="bevel",
    )
    colors = extract_colors(all_data[0], bout_indices)
    collection.set_array(colors)
    collection.set_clim(vmin=vmin, vmax=np.nanmax(colors))
    ax.add_collection(collection)

    cb = plt.colorbar(collection, ax=ax, label=args.var, shrink=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(rm[:, :, 0].min(), rm[:, :, 0].max())
    ax.set_ylim(zm[:, :, 0].min(), zm[:, :, 0].max())
    title = ax.set_title(f"{args.var}  t = {times[0]:.3f}")
    plt.tight_layout()

    def update(frame):
        colors = extract_colors(all_data[frame], bout_indices)
        collection.set_array(colors)
        collection.set_clim(vmin=vmin, vmax=np.nanmax(colors))
        title.set_text(f"{args.var}  t = {times[frame]:.3f}")
        return collection, title

    anim = animation.FuncAnimation(
        fig, update, frames=nt, interval=args.interval, blit=False,
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
