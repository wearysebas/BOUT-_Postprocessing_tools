#!/usr/bin/env python3
"""
BOUT++ 2D animated polygon visualization using Hypnotoad grid cell corners.

Builds polygon patches once from the Hypnotoad corner arrays, then animates
by updating the color array at each time step.

Usage:
    python BOUT_ht_animation.py <grid_file> <sim_results_dir> [options]

Example:
    python BOUT_ht_animation.py \\
        ht_try3.nc /path/to/sim \\
        --var T --zindex 0 --interval 100
"""

import numpy as np
import xarray as xr
import xbout
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection


def build_polygons_ht(grid, MXG):
    """
    Build polygon patches from Hypnotoad grid corner arrays.

    Parameters
    ----------
    grid : xarray.Dataset
        Hypnotoad grid file loaded with xr.open_dataset.
    MXG : int
        Number of radial guard cells on each side to skip.

    Returns
    -------
    patches : list of matplotlib.patches.Polygon
        One patch per physical cell.
    indices : list of (int, int)
        (ix, iy) grid indices corresponding to each patch.
    """
    R_ll = grid["Rxy_corners"].values
    R_lr = grid["Rxy_lower_right_corners"].values
    R_ur = grid["Rxy_upper_right_corners"].values
    R_ul = grid["Rxy_upper_left_corners"].values
    Z_ll = grid["Zxy_corners"].values
    Z_lr = grid["Zxy_lower_right_corners"].values
    Z_ur = grid["Zxy_upper_right_corners"].values
    Z_ul = grid["Zxy_upper_left_corners"].values

    nx, ny = R_ll.shape
    patches = []
    indices = []

    for ix in range(MXG, nx - MXG):
        for iy in range(ny):
            R = [R_ll[ix, iy], R_lr[ix, iy], R_ur[ix, iy], R_ul[ix, iy]]
            Z = [Z_ll[ix, iy], Z_lr[ix, iy], Z_ur[ix, iy], Z_ul[ix, iy]]
            p = matplotlib.patches.Polygon(
                np.array([R, Z]).T, fill=True, closed=True,
            )
            patches.append(p)
            indices.append((ix, iy))

    return patches, indices


def extract_colors(data_2d, indices):
    """Extract color values from a 2D data slice for each polygon.

    Parameters
    ----------
    data_2d : np.ndarray, shape (nx, ny)
        Simulation data at a single time and z index.
    indices : list of (int, int)
        (ix, iy) index pairs as returned by build_polygons_ht.
    """
    return np.array([data_2d[ix, iy] for ix, iy in indices])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BOUT++ 2D animated polygon visualization using Hypnotoad grid"
    )
    parser.add_argument("grid_file", type=str,
                        help="Path to Hypnotoad grid file (.nc)")
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

    grid = xr.open_dataset(args.grid_file, engine="netcdf4")
    sim = xbout.open_boutdataset(
        args.sim_results + "/BOUT.dmp.*.nc",
        gridfilepath=args.grid_file,
        geometry="toroidal",
        keep_yboundaries=False,
    )

    MXG = sim.metadata["MXG"]

    # All time steps: shape (nt, nx, ny) — x includes guard cells
    all_data = sim[args.var].values[:, :, :, args.zindex]
    nt = all_data.shape[0]
    times = sim["t"].values

    # Build polygons once
    patches, indices = build_polygons_ht(grid, MXG)

    # Fixed global min; max updated per frame
    vmin = np.nanmin(all_data)

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 10))

    collection = PatchCollection(
        patches, cmap=args.cmap, edgecolors="face",
        linewidths=0.1, joinstyle="bevel",
    )
    colors = extract_colors(all_data[0], indices)
    collection.set_array(colors)
    collection.set_clim(vmin=vmin, vmax=np.nanmax(colors))
    ax.add_collection(collection)

    cb = plt.colorbar(collection, ax=ax, label=args.var, shrink=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")

    # Axis limits from physical region corners only
    phys = slice(MXG, int(grid["nx"].values) - MXG)
    R_phys = np.concatenate([
        grid["Rxy_corners"].values[phys].ravel(),
        grid["Rxy_lower_right_corners"].values[phys].ravel(),
        grid["Rxy_upper_right_corners"].values[phys].ravel(),
        grid["Rxy_upper_left_corners"].values[phys].ravel(),
    ])
    Z_phys = np.concatenate([
        grid["Zxy_corners"].values[phys].ravel(),
        grid["Zxy_lower_right_corners"].values[phys].ravel(),
        grid["Zxy_upper_right_corners"].values[phys].ravel(),
        grid["Zxy_upper_left_corners"].values[phys].ravel(),
    ])
    ax.set_xlim(R_phys.min(), R_phys.max())
    ax.set_ylim(Z_phys.min(), Z_phys.max())

    title = ax.set_title(f"{args.var}  t = {times[0]:.3f}")
    plt.tight_layout()

    def update(frame):
        colors = extract_colors(all_data[frame], indices)
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
