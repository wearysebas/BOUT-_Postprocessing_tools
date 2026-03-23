#!/usr/bin/env python3
"""
BOUT++ 2D polygon visualization using Hypnotoad grid cell corners.

The Hypnotoad grid stores four corner arrays per cell (Rxy_corners,
Rxy_lower_right_corners, Rxy_upper_left_corners, Rxy_upper_right_corners
and Zxy equivalents). These are used directly to build polygon geometry,
colored by the corresponding simulation data value.

Corner naming convention (per cell (ix, iy)):
    lower = smaller radial index (ix side), upper = larger radial index
    left  = smaller poloidal index (iy side), right = larger poloidal index

    Rxy_corners[ix,iy]              -> lower-left  corner of cell (ix, iy)
    Rxy_lower_right_corners[ix,iy] -> lower-right corner
    Rxy_upper_right_corners[ix,iy] -> upper-right corner
    Rxy_upper_left_corners[ix,iy]  -> upper-left  corner

Usage:
    python BOUT_postprocessing_ht.py <grid_file> <sim_results_dir> [options]

Example:
    python BOUT_postprocessing_ht.py \\
        /path/to/ht_try3.nc /path/to/sim_dir \\
        --var T --time -1 --zindex 0
"""

import numpy as np
import xarray as xr
import xbout
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


def polygon_plot_ht(ax, grid, data, MXG=2, cmap="viridis",
                    vmin=None, vmax=None, colorbar_label=None):
    """
    Create a 2D polygon plot using Hypnotoad grid cell corners.

    Parameters
    ----------
    ax : matplotlib Axes
    grid : xarray.Dataset
        Hypnotoad grid file loaded with xr.open_dataset.
    data : np.ndarray, shape (nx, ny)
        Simulation data including x-guard cells (full nx dimension).
    MXG : int
        Number of radial guard cells on each side to skip.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Color scale limits; auto-detected if None.
    colorbar_label : str or None
        Label for the colorbar.

    Returns
    -------
    PatchCollection
    """
    R_ll = grid["Rxy_corners"].values
    R_lr = grid["Rxy_lower_right_corners"].values
    R_ur = grid["Rxy_upper_right_corners"].values
    R_ul = grid["Rxy_upper_left_corners"].values
    Z_ll = grid["Zxy_corners"].values
    Z_lr = grid["Zxy_lower_right_corners"].values
    Z_ur = grid["Zxy_upper_right_corners"].values
    Z_ul = grid["Zxy_upper_left_corners"].values

    nx, ny = data.shape
    patches = []
    colors = []

    for ix in range(MXG, nx - MXG):
        for iy in range(ny):
            R = [R_ll[ix, iy], R_lr[ix, iy], R_ur[ix, iy], R_ul[ix, iy]]
            Z = [Z_ll[ix, iy], Z_lr[ix, iy], Z_ur[ix, iy], Z_ul[ix, iy]]
            p = matplotlib.patches.Polygon(
                np.array([R, Z]).T, fill=True, closed=True
            )
            patches.append(p)
            colors.append(data[ix, iy])

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

    # Axis limits from physical region corners only
    phys = slice(MXG, nx - MXG)
    R_phys = np.concatenate([
        R_ll[phys].ravel(), R_lr[phys].ravel(),
        R_ur[phys].ravel(), R_ul[phys].ravel(),
    ])
    Z_phys = np.concatenate([
        Z_ll[phys].ravel(), Z_lr[phys].ravel(),
        Z_ur[phys].ravel(), Z_ul[phys].ravel(),
    ])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(R_phys.min(), R_phys.max())
    ax.set_ylim(Z_phys.min(), Z_phys.max())

    return collection


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BOUT++ 2D polygon visualization using Hypnotoad grid cell corners"
    )
    parser.add_argument("grid_file", type=str,
                        help="Path to Hypnotoad grid file (.nc)")
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
    parser.add_argument("--list-vars", action="store_true",
                        help="List available variables in the simulation and exit")

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

    if args.list_vars:
        print("Available simulation variables:")
        for v in sim.data_vars:
            if sim[v].dims == ("t", "x", "theta", "zeta"):
                print(f"  {v}: shape={sim[v].shape}")
        raise SystemExit(0)

    MXG = sim.metadata["MXG"]

    # data shape: (nx, ny) — x includes guard cells, y is physical only
    data = sim[args.var].values[args.time, :, :, args.zindex]

    fig, ax = plt.subplots(figsize=(6, 10))
    polygon_plot_ht(
        ax, grid, data, MXG=MXG,
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
