"""
visualize_ht_grid.py — Visualize hypnotoad BOUT++ grid files.

Unlike the INGRID-based scripts, hypnotoad files do not store rm/zm 5-point
corner arrays.  Instead, they provide four named corner arrays per cell:
  Rxy_corners            (lower-left  in index space: ix-½, iy-½)
  Rxy_upper_left_corners (upper-left  in index space: ix-½, iy+½)
  Rxy_upper_right_corners(upper-right in index space: ix+½, iy+½)
  Rxy_lower_right_corners(lower-right in index space: ix+½, iy-½)

This script uses those four arrays to draw cell polygons, and uses Rxy/Zxy
cell-centers for separatrix and branch-cut lines.

Usage
-----
    python visualize_ht_grid.py <grid_file> [--field FIELD] [--cmap CMAP]
                                            [--no-edges] [--save FILE]

Examples
--------
    python visualize_ht_grid.py ht_test1.nc
    python visualize_ht_grid.py ht_test1.nc --field psixy
    python visualize_ht_grid.py ht_test1.nc --field Bxy --cmap plasma --save Bxy.png
"""

import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import netCDF4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_grid(path):
    """Return an open netCDF4 Dataset."""
    return netCDF4.Dataset(path)


def get_topology_indices(ds):
    """Extract all topology index scalars from the dataset."""
    return {
        "nx":        int(ds.variables["nx"][:]),
        "ny":        int(ds.variables["ny"][:]),
        "ixseps1":   int(ds.variables["ixseps1"][:]),
        "ixseps2":   int(ds.variables["ixseps2"][:]),
        "jyseps1_1": int(ds.variables["jyseps1_1"][:]),
        "jyseps2_1": int(ds.variables["jyseps2_1"][:]),
        "ny_inner":  int(ds.variables["ny_inner"][:]),
        "jyseps1_2": int(ds.variables["jyseps1_2"][:]),
        "jyseps2_2": int(ds.variables["jyseps2_2"][:]),
    }


def build_cell_polygons(ds):
    """
    Build a list of matplotlib Polygon patches, one per grid cell, using
    the four named corner arrays stored in the hypnotoad output file.

    Corner order per cell (traversed counter-clockwise in index space):
        corners → lower_right → upper_right → upper_left → (close)

    Returns
    -------
    patches : list of mpatches.Polygon
        One patch per (ix, iy) cell, in row-major order (ix outer).
    """
    R_ll = ds.variables["Rxy_corners"][:].data            # lower-left
    Z_ll = ds.variables["Zxy_corners"][:].data
    R_ul = ds.variables["Rxy_upper_left_corners"][:].data  # upper-left
    Z_ul = ds.variables["Zxy_upper_left_corners"][:].data
    R_ur = ds.variables["Rxy_upper_right_corners"][:].data # upper-right
    Z_ur = ds.variables["Zxy_upper_right_corners"][:].data
    R_lr = ds.variables["Rxy_lower_right_corners"][:].data # lower-right
    Z_lr = ds.variables["Zxy_lower_right_corners"][:].data

    nx, ny = R_ll.shape
    patches = []
    for ix in range(nx):
        for iy in range(ny):
            verts = np.array([
                [R_ll[ix, iy], Z_ll[ix, iy]],
                [R_lr[ix, iy], Z_lr[ix, iy]],
                [R_ur[ix, iy], Z_ur[ix, iy]],
                [R_ul[ix, iy], Z_ul[ix, iy]],
            ])
            patches.append(mpatches.Polygon(verts, closed=True))
    return patches, nx, ny


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_grid_only(ds, topo, ax, edgecolor="black", linewidth=0.3):
    """Draw cell outlines with no fill."""
    print("Building cell polygons...")
    patches, nx, ny = build_cell_polygons(ds)
    col = PatchCollection(patches, facecolor="none",
                          edgecolor=edgecolor, linewidth=linewidth)
    ax.add_collection(col)
    print(f"  Drew {nx * ny} cell outlines.")


def plot_field(ds, topo, field_name, ax, cmap="inferno",
               show_edges=True, edge_linewidth=0.15):
    """Fill cells with colour according to a 2D field, return the collection."""
    if field_name not in ds.variables:
        raise KeyError(
            f"Field '{field_name}' not found in the grid file.\n"
            f"Available 2D fields: {[k for k, v in ds.variables.items() if len(v.shape)==2]}"
        )
    values = ds.variables[field_name][:].data.flatten(order="C")  # ix outer, iy inner

    print(f"Building cell polygons for field '{field_name}'...")
    patches, nx, ny = build_cell_polygons(ds)

    edgecol  = "k" if show_edges else "none"
    col = PatchCollection(patches, cmap=cmap,
                          edgecolor=edgecol, linewidth=edge_linewidth)
    col.set_array(values)
    ax.add_collection(col)
    plt.colorbar(col, ax=ax, shrink=0.8, label=field_name)
    print(f"  Drew {nx * ny} filled cells.")
    return col


def plot_separatrix_and_cuts(ds, topo, ax):
    """
    Overlay separatrix (ixseps1), branch cuts (jyseps*) and inner target
    line (ny_inner) using cell-centre coordinates.
    """
    R = ds.variables["Rxy"][:].data
    Z = ds.variables["Zxy"][:].data

    ix1  = topo["ixseps1"]
    ix2  = topo["ixseps2"]
    jy11 = topo["jyseps1_1"]
    jy21 = topo["jyseps2_1"]
    ny_i = topo["ny_inner"]
    jy12 = topo["jyseps1_2"]
    jy22 = topo["jyseps2_2"]
    nx   = topo["nx"]
    ny   = topo["ny"]

    # Clamp indices to valid range
    ix1c  = min(ix1,  nx - 1)
    ix2c  = min(ix2,  nx - 1)

    # -- Separatrix(ces) --
    ax.plot(R[ix1c, :], Z[ix1c, :],
            color="magenta", lw=1.5, label=f"ixseps1={ix1}", zorder=3)
    if ix2c != ix1c and ix2 < nx:
        ax.plot(R[ix2c, :], Z[ix2c, :],
                color="yellow", lw=1.5, label=f"ixseps2={ix2}", zorder=3)

    # -- Branch cuts (radial slices at poloidal junctions) --
    ax.plot(R[:, jy11], Z[:, jy11],
            color="gold", lw=1.2, ls="--", label=f"jyseps1_1={jy11}", zorder=3)
    ax.plot(R[:, jy21], Z[:, jy21],
            color="limegreen", lw=1.2, ls="--", label=f"jyseps2_1={jy21}", zorder=3)
    ax.plot(R[:, ny_i], Z[:, ny_i],
            color="cyan", lw=1.2, ls="--", label=f"ny_inner={ny_i}", zorder=3)
    ax.plot(R[:, jy12], Z[:, jy12],
            color="cornflowerblue", lw=1.2, ls="--", label=f"jyseps1_2={jy12}", zorder=3)
    ax.plot(R[:, jy22], Z[:, jy22],
            color="tomato", lw=1.2, ls="--", label=f"jyseps2_2={jy22}", zorder=3)


def plot_wall(ds, ax):
    """Draw the machine wall contour if present in the file."""
    if "closed_wall_R" not in ds.variables:
        return
    Rw = ds.variables["closed_wall_R"][:].data
    Zw = ds.variables["closed_wall_Z"][:].data
    ax.plot(Rw, Zw, color="dimgray", lw=2.0, label="wall", zorder=4)


def set_axis_style(ax, ds, title=""):
    """Set axis limits, labels, aspect ratio."""
    R = ds.variables["Rxy"][:].data
    Z = ds.variables["Zxy"][:].data
    margin_R = 0.05 * (R.max() - R.min())
    margin_Z = 0.05 * (Z.max() - Z.min())
    ax.set_xlim(R.min() - margin_R, R.max() + margin_R)
    ax.set_ylim(Z.min() - margin_Z, Z.max() + margin_Z)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a hypnotoad BOUT++ grid file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("grid_file", help="Path to hypnotoad .nc grid file")
    parser.add_argument(
        "--field", default=None,
        help="Name of a 2D field to colour cells by (e.g. psixy, Bxy, Bpxy). "
             "If omitted, only cell outlines are drawn.",
    )
    parser.add_argument(
        "--cmap", default="inferno",
        help="Matplotlib colormap for --field (default: inferno)",
    )
    parser.add_argument(
        "--no-edges", action="store_true",
        help="Hide cell edge lines when a field is plotted",
    )
    parser.add_argument(
        "--save", default=None, metavar="FILE",
        help="Save figure to FILE instead of displaying interactively",
    )
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    ds   = load_grid(args.grid_file)
    topo = get_topology_indices(ds)

    print("Grid file:", args.grid_file)
    print(f"  nx={topo['nx']}, ny={topo['ny']}")
    print(f"  ixseps1={topo['ixseps1']}, ixseps2={topo['ixseps2']}")
    print(f"  jyseps1_1={topo['jyseps1_1']}, jyseps2_1={topo['jyseps2_1']}, "
          f"ny_inner={topo['ny_inner']}, jyseps1_2={topo['jyseps1_2']}, "
          f"jyseps2_2={topo['jyseps2_2']}")

    fig, ax = plt.subplots(figsize=(7, 10))

    if args.field is None:
        plot_grid_only(ds, topo, ax, edgecolor="steelblue", linewidth=0.3)
        title = f"Hypnotoad grid — {args.grid_file}"
    else:
        plot_field(ds, topo, args.field, ax,
                   cmap=args.cmap,
                   show_edges=not args.no_edges,
                   edge_linewidth=0.15)
        title = f"{args.field} — {args.grid_file}"

    plot_separatrix_and_cuts(ds, topo, ax)
    plot_wall(ds, ax)
    set_axis_style(ax, ds, title=title)

    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()

    ds.close()


if __name__ == "__main__":
    main()
