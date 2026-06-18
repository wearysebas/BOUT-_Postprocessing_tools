#!/usr/bin/env python3
"""
Visualize BOUT++ cylindrical coordinates (Rxy, Zxy) from the grid file
on the R-Z plane.  Useful for designing initial conditions with PR 3222's
input:grid_variables feature, e.g.:

    [input:grid_variables]
    Rxy = field2d
    Zxy = field2d

    [T]
    function = gauss(Rxy - 1.5, 0.1) * gauss(Zxy + 1.0, 0.1)

Usage:
    python BOUT_cyl_coordinates_visualizer.py <dump_dir>
    python BOUT_cyl_coordinates_visualizer.py <dump_dir> --highlight-R 1.5 --highlight-Z -1.0
    python BOUT_cyl_coordinates_visualizer.py <dump_dir> --grid /path/to/grid.nc
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
        description="Plot BOUT++ Rxy and Zxy from the grid file on the R-Z plane."
    )
    parser.add_argument("dump_dir",
                        help="Directory containing BOUT.inp (for grid file auto-detection)")
    parser.add_argument("--grid", type=str, default=None,
                        help="Path to grid file (.nc). Auto-detected from BOUT.inp if omitted.")
    parser.add_argument("--cmap", type=str, default="inferno",
                        help="Colormap for the Rxy plot (default: inferno)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to this path instead of displaying interactively")
    parser.add_argument("--highlight-R", type=float, default=None,
                        help="Highlight cells near this R value [m]")
    parser.add_argument("--highlight-Z", type=float, default=None,
                        help="Highlight cells near this Z value [m]")
    parser.add_argument("--tol-R", type=float, default=0.03,
                        help="Tolerance for R highlight matching [m] (default: 0.03)")
    parser.add_argument("--tol-Z", type=float, default=0.03,
                        help="Tolerance for Z highlight matching [m] (default: 0.03)")

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
    Rxy = grid["Rxy"].values   # (nx, ny)
    Zxy = grid["Zxy"].values
    nx = int(grid["nx"].values)
    ny = int(grid["ny"].values)
    grid.close()
    print(f"Grid: nx={nx}, ny={ny}  ({grid_path})")
    print(f"  Rxy range: [{Rxy.min():.4f}, {Rxy.max():.4f}] m")
    print(f"  Zxy range: [{Zxy.min():.4f}, {Zxy.max():.4f}] m")

    # --- flatten (skip x wall guards) ---
    R = Rxy[MXG:nx - MXG, :].ravel()
    Z = Zxy[MXG:nx - MXG, :].ravel()

    # --- highlight masks ---
    hR = args.highlight_R
    hZ = args.highlight_Z
    tol_R = args.tol_R
    tol_Z = args.tol_Z

    if hR is not None:
        nearest_R = R[np.argmin(np.abs(R - hR))]
        mask_R = np.abs(R - hR) < tol_R
        if not mask_R.any():
            print(f"  WARNING: no cells within tol_R={tol_R} of R={hR}. "
                  f"Nearest R value: {nearest_R:.6f} (delta={abs(nearest_R-hR):.6f}). "
                  f"Snapping to nearest.")
            mask_R = np.abs(R - nearest_R) < 1e-9
    else:
        mask_R = None

    if hZ is not None:
        nearest_Z = Z[np.argmin(np.abs(Z - hZ))]
        mask_Z = np.abs(Z - hZ) < tol_Z
        if not mask_Z.any():
            print(f"  WARNING: no cells within tol_Z={tol_Z} of Z={hZ}. "
                  f"Nearest Z value: {nearest_Z:.6f} (delta={abs(nearest_Z-hZ):.6f}). "
                  f"Snapping to nearest.")
            mask_Z = np.abs(Z - nearest_Z) < 1e-9
            hZ = nearest_Z
    else:
        mask_Z = None

    # --- layout ---
    ncols = 3 if (hR is not None and hZ is not None) else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 10))
    ax1, ax2 = axes[0], axes[1]

    # --- Rxy plot ---
    sc1 = ax1.scatter(R, Z, c=R, cmap=args.cmap, s=2, linewidths=0, rasterized=True)
    plt.colorbar(sc1, ax=ax1, shrink=0.8, label="Rxy [m]")
    if mask_R is not None:
        ax1.scatter(R[mask_R], Z[mask_R], c="red", s=8, linewidths=0, zorder=5,
                    label=f"R={hR}")
        ax1.legend(fontsize=8)
    ax1.set_title("Rxy (major radius)")

    # --- Zxy plot ---
    sc2 = ax2.scatter(R, Z, c=Z, cmap="viridis", s=2, linewidths=0, rasterized=True)
    plt.colorbar(sc2, ax=ax2, shrink=0.8, label="Zxy [m]")
    if mask_Z is not None:
        ax2.scatter(R[mask_Z], Z[mask_Z], c="red", s=8, linewidths=0, zorder=5,
                    label=f"Z={hZ}")
        ax2.legend(fontsize=8)
    ax2.set_title("Zxy (vertical position)")

    # --- third plot: cells matching BOTH R and Z ---
    if ncols == 3:
        ax3 = axes[2]
        mask_both = mask_R & mask_Z
        ax3.scatter(R, Z, c="lightgrey", s=2, linewidths=0, rasterized=True)
        ax3.scatter(R[mask_both], Z[mask_both], c="red", s=20, linewidths=0, zorder=5,
                    label=f"R={hR}, Z={hZ}")
        n_hits = mask_both.sum()
        ax3.set_title(f"Cells at (R={hR}, Z={hZ})  [{n_hits} hits]")
        ax3.legend(fontsize=8)
        print(f"  Highlight (R={hR}, Z={hZ}): {n_hits} cells (tol_R={tol_R}, tol_Z={tol_Z})")

    # Print copy-pasteable BOUT.inp snippet
    if hR is not None or hZ is not None:
        # Determine centers from whichever masks are active
        if mask_R is not None and mask_Z is not None:
            mask_both = mask_R & mask_Z
        elif mask_R is not None:
            mask_both = mask_R
        else:
            mask_both = mask_Z

        if mask_both.any():
            R_center = R[mask_both].mean()
            Z_center = Z[mask_both].mean()
        else:
            R_center = hR if hR is not None else R.mean()
            Z_center = hZ if hZ is not None else Z.mean()

        R_str = f"{R_center:.6f}"
        Z_str = f"{Z_center:.6f}"

        print(f"\nCopy into BOUT.inp:\n")
        print(f"[input:grid_variables]")
        print(f"Rxy = field2d")
        print(f"Zxy = field2d")
        print()
        print(f"[T]")
        print(f"scale = 1.0")
        if hR is not None and hZ is not None:
            print(f"function = gauss(Rxy - ({R_str}), 0.05) * gauss(Zxy - ({Z_str}), 0.05)")
        elif hR is not None:
            print(f"function = gauss(Rxy - ({R_str}), 0.05)")
        else:
            print(f"function = gauss(Zxy - ({Z_str}), 0.05)")
        print()
        print(f"# Using named constants (add R_0/Z_0 to the grid file):")
        print(f"# [input:grid_variables]")
        print(f"# R_0 = boutreal    # = {R_str}")
        print(f"# Z_0 = boutreal    # = {Z_str}")
        if hR is not None and hZ is not None:
            print(f"# function = gauss(Rxy - R_0, 0.05) * gauss(Zxy - Z_0, 0.05)")
        elif hR is not None:
            print(f"# function = gauss(Rxy - R_0, 0.05)")
        else:
            print(f"# function = gauss(Zxy - Z_0, 0.05)")

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
