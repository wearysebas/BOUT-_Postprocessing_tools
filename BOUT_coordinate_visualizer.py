#!/usr/bin/env python3
"""
Visualize BOUT++ (x, y) coordinate fields from BOUT_coord.dmp.*.nc files
on the R-Z plane.

Usage:
    python BOUT_coordinate_visualizer.py <dump_dir> [options]

Example:
    python BOUT_coordinate_visualizer.py /path/to/Corrected_y0_West_PFR
    python BOUT_coordinate_visualizer.py /path/to/Corrected_y0_West_PFR --output coords.png
    python BOUT_coordinate_visualizer.py /path/to/Corrected_y0_West_PFR --grid /path/to/grid.nc
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

MXG = 2
MYG = 2


def sorted_dump_files(dump_dir):
    files = glob.glob(os.path.join(dump_dir, "BOUT_coord.dmp.*.nc"))
    return sorted(files, key=lambda p: int(re.search(r'\.(\d+)\.nc$', p).group(1)))


def find_grid_file(dump_dir):
    """Read grid filename from BOUT.inp and search for it in nearby directories."""
    inp_path = os.path.join(dump_dir, "BOUT.inp")
    if not os.path.exists(inp_path):
        return None

    grid_name = None
    in_mesh = False
    with open(inp_path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("["):
                in_mesh = line.lower().startswith("[mesh]")
            if in_mesh and "=" in line:
                key = line.split("=")[0].strip().lower()
                if key == "file":
                    m = re.search(r'=\s*"?([^\s"#]+)"?', line)
                    if m:
                        grid_name = m.group(1)
                        break

    if grid_name is None:
        return None

    # Search dump_dir, parent, and siblings
    parent = os.path.dirname(os.path.abspath(dump_dir))
    for root, _dirs, fnames in os.walk(parent):
        if grid_name in fnames:
            return os.path.join(root, grid_name)

    return None


def stitch_field(files, field, grid_nx, grid_ny):
    """
    Assemble a 2D (grid_nx, grid_ny) field from per-processor dump files
    opened with xr.open_dataset.

    BOUT++ convention:
      x — wall guards kept, inter-proc guards stripped
      y — all guards stripped (local[MYG:-MYG])
    """
    ds0 = xr.open_dataset(files[0], engine="netcdf4")
    lnx, lny = ds0[field].shape[:2]
    ds0.close()

    inner_y = lny - 2 * MYG
    inner_x = lnx - 2 * MXG
    nype = grid_ny // inner_y
    nxpe = len(files) // nype

    full = np.full((grid_nx, grid_ny), np.nan)

    for ix_proc in range(nxpe):
        for iy_proc in range(nype):
            ds = xr.open_dataset(files[iy_proc * nxpe + ix_proc], engine="netcdf4")
            data = ds[field].values[:, :, 0]   # take z=0 slice
            ds.close()

            if nxpe == 1:
                xl, gx0, gxn = slice(None), 0, lnx
            elif ix_proc == 0:
                xl, gx0, gxn = slice(0, lnx - MXG), 0, lnx - MXG
            elif ix_proc == nxpe - 1:
                xl, gx0, gxn = slice(MXG, lnx), ix_proc * inner_x, lnx - MXG
            else:
                xl, gx0, gxn = slice(MXG, lnx - MXG), MXG + ix_proc * inner_x, inner_x

            gy0 = iy_proc * inner_y
            full[gx0:gx0 + gxn, gy0:gy0 + inner_y] = data[xl, MYG:lny - MYG]

    return full


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot BOUT++ x and y coordinates on the R-Z plane."
    )
    parser.add_argument(
        "dump_dir",
        help="Directory containing the BOUT_coord.dmp.*.nc files",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Path to BOUT++ grid file (.nc). Auto-detected from BOUT.inp if omitted.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        help="Colormap for the x coordinate plot (default: inferno)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save figure to this path instead of displaying interactively",
    )
    parser.add_argument(
        "--highlight-x", type=float, default=None,
        help="Highlight cells near this x coordinate value",
    )
    parser.add_argument(
        "--highlight-y", type=float, default=None,
        help="Highlight cells near this y coordinate value",
    )
    parser.add_argument(
        "--tol-x", type=float, default=0.01,
        help="Tolerance for x highlight matching (default: 0.01, x range ~0-0.5)",
    )
    parser.add_argument(
        "--tol-y", type=float, default=0.1,
        help="Tolerance for y highlight matching (default: 0.1, y range 0-2pi)",
    )

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    # --- dump files ---
    files = sorted_dump_files(args.dump_dir)
    if not files:
        print(f"No BOUT_coord.dmp.*.nc files found in {args.dump_dir}")
        sys.exit(1)
    print(f"Found {len(files)} dump files")

    # --- grid file ---
    grid_path = args.grid or find_grid_file(args.dump_dir)
    if grid_path is None or not os.path.exists(grid_path):
        print("Grid file not found. Provide it with --grid.")
        sys.exit(1)

    grid = xr.open_dataset(grid_path, engine="netcdf4")
    Rxy = grid["Rxy"].values   # (nx, ny)
    Zxy = grid["Zxy"].values
    nx  = int(grid["nx"].values)
    ny  = int(grid["ny"].values)
    grid.close()
    print(f"Grid: nx={nx}, ny={ny}  ({grid_path})")

    # --- stitch coordinate fields ---
    x_field = stitch_field(files, "x", nx, ny)
    y_field = stitch_field(files, "y", nx, ny)

    # --- flatten to plottable arrays (skip x wall guards) ---
    R  = Rxy[MXG:nx - MXG, :].ravel()
    Z  = Zxy[MXG:nx - MXG, :].ravel()
    xv = x_field[MXG:nx - MXG, :].ravel()
    yv = y_field[MXG:nx - MXG, :].ravel()

    # --- highlight masks ---
    hx = args.highlight_x
    hy = args.highlight_y
    tol_x = args.tol_x
    tol_y = args.tol_y

    if hx is not None:
        nearest_x = xv[np.argmin(np.abs(xv - hx))]
        mask_x = np.abs(xv - hx) < tol_x
        if not mask_x.any():
            print(f"  WARNING: no cells within tol_x={tol_x} of x={hx}. "
                  f"Nearest x value: {nearest_x:.6f} (delta={abs(nearest_x-hx):.6f}). "
                  f"Snapping to nearest.")
            mask_x = np.abs(xv - nearest_x) < 1e-9
    else:
        mask_x = None

    if hy is not None:
        nearest_y = yv[np.argmin(np.abs(yv - hy))]
        mask_y = np.abs(yv - hy) < tol_y
        if not mask_y.any():
            print(f"  WARNING: no cells within tol_y={tol_y} of y={hy}. "
                  f"Nearest y value: {nearest_y:.6f} (delta={abs(nearest_y-hy):.6f}). "
                  f"Snapping to nearest.")
            mask_y = np.abs(yv - nearest_y) < 1e-9
            hy = nearest_y  # update for labels
    else:
        mask_y = None

    # --- layout: 2 panels normally, 3 if highlighting both x and y ---
    ncols = 3 if (hx is not None and hy is not None) else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 10))
    ax1, ax2 = axes[0], axes[1]

    # --- x coordinate plot ---
    sc1 = ax1.scatter(R, Z, c=xv, cmap=args.cmap, s=2, linewidths=0, rasterized=True)
    plt.colorbar(sc1, ax=ax1, shrink=0.8, label="x (normalised radial)")
    if mask_x is not None:
        ax1.scatter(R[mask_x], Z[mask_x], c="red", s=8, linewidths=0, zorder=5,
                    label=f"x={hx}")
        ax1.legend(fontsize=8)
    ax1.set_title("BOUT++ x coordinate")

    # --- y coordinate plot ---
    sc2 = ax2.scatter(R, Z, c=yv, cmap="viridis", s=2, linewidths=0, rasterized=True)
    plt.colorbar(sc2, ax=ax2, shrink=0.8, label="y (poloidal angle)")
    if mask_y is not None:
        ax2.scatter(R[mask_y], Z[mask_y], c="red", s=8, linewidths=0, zorder=5,
                    label=f"y={hy}")
        ax2.legend(fontsize=8)
    ax2.set_title("BOUT++ y coordinate")

    # --- third plot: cells matching BOTH x and y ---
    if ncols == 3:
        ax3 = axes[2]
        mask_both = mask_x & mask_y
        # grey background of all cells
        ax3.scatter(R, Z, c="lightgrey", s=2, linewidths=0, rasterized=True)
        ax3.scatter(R[mask_both], Z[mask_both], c="red", s=20, linewidths=0, zorder=5,
                    label=f"x={hx}, y={hy}")
        n_hits = mask_both.sum()
        ax3.set_title(f"Cells at (x={hx}, y={hy})  [{n_hits} hits]")
        ax3.legend(fontsize=8)
        print(f"  Highlight (x={hx}, y={hy}): {n_hits} cells (tol_x={tol_x}, tol_y={tol_y})")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()
