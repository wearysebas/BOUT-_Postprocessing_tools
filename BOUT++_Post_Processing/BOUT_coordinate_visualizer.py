#!/usr/bin/env python3
"""
Visualize BOUT++ field-aligned (x, y) coordinates on the R-Z plane,
using Rxy/Zxy from the grid file for spatial layout and computing the
x (normalised radial) and y (poloidal angle) coordinates analytically
from the grid indices.

Useful for designing initial conditions that use the built-in BOUT++
x and y expression variables, e.g.:

    [T]
    function = gauss(x - 0.5, 0.05) * gauss(y - 3.14159, 0.2)

Usage:
    python BOUT_coordinate_visualizer.py <dump_dir>
    python BOUT_coordinate_visualizer.py <dump_dir> --highlight-x 0.5 --highlight-y 3.14
    python BOUT_coordinate_visualizer.py <dump_dir> --grid /path/to/grid.nc
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

MXG = 2
TWOPI = 2.0 * np.pi


def find_grid_file(dump_dir):
    """Read grid filename from BOUT.inp [mesh] section and find it nearby."""
    inp_path = os.path.join(dump_dir, "BOUT.inp")
    if not os.path.exists(inp_path):
        return None
    in_mesh = False
    with open(inp_path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("["):
                in_mesh = line.lower().startswith("[mesh]")
            if in_mesh and "=" in line and line.split("=")[0].strip().lower() == "file":
                m = re.search(r'=\s*"?([^\s"#]+)"?', line)
                if m:
                    name = m.group(1)
                    parent = os.path.dirname(os.path.abspath(dump_dir))
                    for root, _, fnames in os.walk(parent):
                        if name in fnames:
                            return os.path.join(root, name)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot BOUT++ field-aligned x and y coordinates on the R-Z plane."
    )
    parser.add_argument("dump_dir",
                        help="Directory containing BOUT.inp (for grid file auto-detection)")
    parser.add_argument("--grid", type=str, default=None,
                        help="Path to grid file (.nc). Auto-detected from BOUT.inp if omitted.")
    parser.add_argument("--cmap", type=str, default="inferno",
                        help="Colormap for the x coordinate plot (default: inferno)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to this path instead of displaying interactively")
    parser.add_argument("--highlight-x", type=float, default=None,
                        help="Highlight cells near this x value (normalised radial, 0..1)")
    parser.add_argument("--highlight-y", type=float, default=None,
                        help="Highlight cells near this y value (poloidal angle, 0..2pi)")
    parser.add_argument("--tol-x", type=float, default=0.01,
                        help="Tolerance for x highlight matching (default: 0.01)")
    parser.add_argument("--tol-y", type=float, default=0.1,
                        help="Tolerance for y highlight matching (default: 0.1)")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    # --- grid file ---
    grid_path = args.grid or find_grid_file(args.dump_dir)
    if grid_path is None or not os.path.exists(grid_path):
        print("Grid file not found. Provide it with --grid.")
        sys.exit(1)

    grid = xr.open_dataset(grid_path, engine="netcdf4")
    Rxy = grid["Rxy"].values   # (nx, ny)  nx includes MXG guards, ny has no y-guards
    Zxy = grid["Zxy"].values
    nx = int(grid["nx"].values)
    ny = int(grid["ny"].values)
    grid.close()
    print(f"Grid: nx={nx}, ny={ny}  ({grid_path})")

    # --- compute BOUT++ field-aligned coordinates analytically ---
    # Standard BOUT++ formula (cell-centred):
    #   x = (i_global - MXG + 0.5) / (nx_global - 2*MXG)
    #   y = 2*pi * (j_global - MYG + 0.5) / (ny_global - 2*MYG)
    # The grid file's nx includes x-guards, while its ny is the inner poloidal
    # count (BOUT++ adds y-guards at runtime).  So:
    nx_inner = nx - 2 * MXG
    i_idx = np.arange(nx)
    j_idx = np.arange(ny)
    x_1d = (i_idx - MXG + 0.5) / nx_inner          # negative in radial guards, 0..1 inside
    y_1d = TWOPI * (j_idx + 0.5) / ny              # 0..2pi across the (guard-free) poloidal domain
    x_field = np.broadcast_to(x_1d[:, None], (nx, ny))
    y_field = np.broadcast_to(y_1d[None, :], (nx, ny))

    # --- flatten (skip x wall guards) ---
    R  = Rxy[MXG:nx - MXG, :].ravel()
    Z  = Zxy[MXG:nx - MXG, :].ravel()
    xv = x_field[MXG:nx - MXG, :].ravel()
    yv = y_field[MXG:nx - MXG, :].ravel()
    print(f"  x range: [{xv.min():.4f}, {xv.max():.4f}]")
    print(f"  y range: [{yv.min():.4f}, {yv.max():.4f}]")

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
            hy = nearest_y
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
                    label=f"y={hy:.4f}")
        ax2.legend(fontsize=8)
    ax2.set_title("BOUT++ y coordinate")

    # --- third plot: cells matching BOTH x and y ---
    if ncols == 3:
        ax3 = axes[2]
        mask_both = mask_x & mask_y
        ax3.scatter(R, Z, c="lightgrey", s=2, linewidths=0, rasterized=True)
        ax3.scatter(R[mask_both], Z[mask_both], c="red", s=20, linewidths=0, zorder=5,
                    label=f"x={hx}, y={hy:.4f}")
        n_hits = mask_both.sum()
        ax3.set_title(f"Cells at (x={hx}, y={hy:.4f})  [{n_hits} hits]")
        ax3.legend(fontsize=8)
        print(f"  Highlight (x={hx}, y={hy}): {n_hits} cells (tol_x={tol_x}, tol_y={tol_y})")

    # --- copy-pasteable BOUT.inp snippet ---
    if hx is not None or hy is not None:
        if mask_x is not None and mask_y is not None:
            mask_sel = mask_x & mask_y
        elif mask_x is not None:
            mask_sel = mask_x
        else:
            mask_sel = mask_y

        if mask_sel.any():
            x_center = xv[mask_sel].mean()
            y_center = yv[mask_sel].mean()
        else:
            x_center = hx if hx is not None else xv.mean()
            y_center = hy if hy is not None else yv.mean()

        x_str = f"{x_center:.6f}"
        y_str = f"{y_center:.6f}"

        print(f"\nCopy into BOUT.inp:\n")
        print(f"[T]")
        print(f"scale = 1.0")
        if hx is not None and hy is not None:
            print(f"function = gauss(x - ({x_str}), 0.05) * gauss(y - ({y_str}), 0.2)")
        elif hx is not None:
            print(f"function = gauss(x - ({x_str}), 0.05)")
        else:
            print(f"function = gauss(y - ({y_str}), 0.2)")

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
