#!/usr/bin/env python3
"""
Hermes-3 coordinate visualiser: plots BOUT++ field-aligned (x, y)
coordinates on the R-Z plane (using Rxy/Zxy from the grid file) and
generates ready-to-paste Hermes-3 input snippets for initial-condition
sources bounded by the highlighted region.

Usage:
    python Hermes-coord-visualizer.py <dump_dir> --highlight-x 0.35 --tol-x 0.35
    python Hermes-coord-visualizer.py <dump_dir> --highlight-x 0.35 --tol-x 0.35 \\
        --core_volume 0.29689 --power_in 1.0
    python Hermes-coord-visualizer.py <dump_dir> --highlight-x 0.35 --tol-x 0.35 \\
        --no-source_only_in_core
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


def heaviside_factor(coord, lo, hi, domain_hi):
    """Build 'H(coord - lo) * H(hi - coord)', simplifying edge cases.

    - lo at the domain start (0)        -> H(coord)
    - hi at or past the domain end      -> drop the upper factor
    """
    parts = []
    if lo > 1e-9:
        parts.append(f"H({coord}-{lo:.6g})")
    else:
        parts.append(f"H({coord})")
    if hi < domain_hi - 1e-9:
        parts.append(f"H({hi:.6g}-{coord})")
    return " * ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise BOUT++ x/y coords and emit Hermes-3 source snippet."
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
                        help="Centre of the x highlight band (normalised radial, 0..1)")
    parser.add_argument("--highlight-y", type=float, default=None,
                        help="Centre of the y highlight band (poloidal angle, 0..2pi)")
    parser.add_argument("--tol-x", type=float, default=0.01,
                        help="Half-width of the x highlight band (default: 0.01)")
    parser.add_argument("--tol-y", type=float, default=0.1,
                        help="Half-width of the y highlight band (default: 0.1)")
    parser.add_argument("--core_volume", type=float, default=0.29689,
                        help="core_volume value written into the snippet [m^3] (default: 0.29689)")
    parser.add_argument("--power_in", type=float, default=1.0,
                        help="power_in value written into the snippet [MW] (default: 1.0)")
    parser.add_argument("--source_only_in_core", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Set source_only_in_core in the snippet (default: True). "
                             "Use --no-source_only_in_core to disable.")

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

    # --- BOUT++ field-aligned coordinates (cell-centred) ---
    nx_inner = nx - 2 * MXG
    i_idx = np.arange(nx)
    j_idx = np.arange(ny)
    x_1d = (i_idx - MXG + 0.5) / nx_inner          # 0..1 across the inner radial domain
    y_1d = TWOPI * (j_idx + 0.5) / ny              # 0..2pi across the poloidal domain
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

    # --- Hermes-3 snippet ---
    if hx is not None or hy is not None:
        factors = []
        if hx is not None:
            x_lo = max(hx - tol_x, 0.0)
            x_hi = min(hx + tol_x, 1.0)
            factors.append(heaviside_factor("x", x_lo, x_hi, 1.0))
        if hy is not None:
            y_lo = max(hy - tol_y, 0.0)
            y_hi = min(hy + tol_y, TWOPI)
            factors.append(heaviside_factor("y", y_lo, y_hi, TWOPI))
        factors.append("pressure_source")

        print("\nCopy into BOUT.inp:\n")
        print(f"core_volume = {args.core_volume:g}                 # m^3")
        print(f"power_in = {args.power_in:g}                       # MW")
        print(f"pressure_source = power_in * 1e6 * 2/3 / core_volume")
        print(f"source = " + " * ".join(factors))
        print(f"source_only_in_core = {'true' if args.source_only_in_core else 'false'}")

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
