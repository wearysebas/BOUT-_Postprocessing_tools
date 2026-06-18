#!/usr/bin/env python3
"""
BOUT++ 2D polygon visualization using grid cell corners from gridue data (rm, zm).

Iterates over the raw gridue cells and uses their actual corner coordinates
for polygon geometry, then colors each cell by the corresponding BOUT
simulation data value.

Usage:
    python BOUT_postprocessing_sdtools.py <grid_file> <sim_results_dir> [options]

Example:
    python BOUT_postprocessing_sdtools.py \
        1st_try/gridue_try6_bout_from_in.grd.nc 1st_try \
        --var T --time -1 --zindex 0
"""

import numpy as np
import xarray as xr
import xbout
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection


def polygon_plot(ax, rm, zm, data, ny_inner, cmap="viridis",
                 vmin=None, vmax=None, colorbar_label=None):
    """
    Create a 2D polygon plot by iterating over raw gridue cells.

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

    data has shape (bout_nx, bout_ny) = (nx, ny) with x guard cells kept.
    The gridue mapping naturally only accesses bout_i = j+1 = 2..Ny-2, which
    are the physical cells; x guard cells are never reached.

    Mapping from gridue (i, j) to BOUT (bout_i, bout_j):
      Radial:   gridue j=1..Ny-2  -> bout_i = j + 1
      Poloidal: gridue i=1..Nx-2  -> bout_j depends on ny_inner:
                  post_step1 = i - 1
                  if post_step1 < ny_inner:        bout_j = post_step1
                  if post_step1 == ny_inner:       guard cell -> nearest neighbour (ny_inner-1)
                  if post_step1 == ny_inner+1:     guard cell -> nearest neighbour (ny_inner)
                  if post_step1 > ny_inner+1:     bout_j = post_step1 - 2
    """
    Nx_gridue = rm.shape[0]
    Ny_gridue = rm.shape[1]
    nx_data, bout_ny = data.shape

    patches = []
    colors = []
    idx = [np.array([1, 2, 4, 3, 1])]

    for i in range(Nx_gridue):
        for j in range(Ny_gridue):
            # Map gridue (i, j) -> BOUT (bout_i, bout_j)

            # Radial: skip boundary cells (j=0 and j=Ny-1)
            if j == 0 or j == Ny_gridue - 1:
                continue
            bout_i = j + 1  # gridue j=1 -> bout_i=2, ..., j=Ny-2 -> bout_i=Ny-1

            # Poloidal: skip boundary cells (i=0 and i=Nx-1)
            if i == 0 or i == Nx_gridue - 1:
                continue
            post_step1 = i - 1
            if post_step1 < ny_inner:
                bout_j = post_step1
            elif post_step1 == ny_inner:
                # X-point guard cell: use nearest neighbour on the inner side
                bout_j = ny_inner - 1
            elif post_step1 == ny_inner + 1:
                # X-point guard cell: use nearest neighbour on the outer side
                bout_j = ny_inner
            else:
                bout_j = post_step1 - 2

            # Bounds check
            if bout_i < 0 or bout_i >= nx_data or bout_j < 0 or bout_j >= bout_ny:
                continue

            # Build polygon from actual gridue corners
            p = matplotlib.patches.Polygon(
                np.concatenate((rm[i][j][idx], zm[i][j][idx])).reshape(2, 5).T,
                fill=True,
                closed=True,
            )
            patches.append(p)
            colors.append(data[bout_i, bout_j])

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

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(rm[:, :, 0].min(), rm[:, :, 0].max())
    ax.set_ylim(zm[:, :, 0].min(), zm[:, :, 0].max())

    return collection


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BOUT++ 2D polygon visualization using gridue cell corners"
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

    # Simulation data: keep full x dimension (guard cells included).
    # The gridue mapping bout_i = j+1 naturally lands on physical cells only
    # (bout_i = 2..Ny_gridue-2), so guard cells are never accessed.
    data = sim[args.var].values[args.time, :, :, args.zindex]

    # Raw gridue cell corners from grid file
    rm = grid["rm"].values  # (Nx_gridue, Ny_gridue, 5)
    zm = grid["zm"].values

    # Plot
    fig, ax = plt.subplots(figsize=(6, 10))
    polygon_plot(ax, rm, zm, data, ny_inner,
                 cmap=args.cmap, colorbar_label=args.var)
    ax.set_title(f"{args.var} (t={args.time}, z={args.zindex})")
    plt.tight_layout()
    plt.show()
