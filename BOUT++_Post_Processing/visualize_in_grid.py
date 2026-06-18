"""
visualize_in_grid.py — Visualize INGRID-generated BOUT++ grid files.

INGRID BOUT++ files store the cell-corner geometry in two arrays:
  rm   shape (n_pol, n_rad, 5)   R-coordinates
  zm   shape (n_pol, n_rad, 5)   Z-coordinates

The 5 corner indices are:
  k=0  cell centre
  k=1  NW corner  (high-y, low-x  in poloidal-radial space)
  k=2  SW corner  (low-y,  low-x)
  k=3  NE corner  (high-y, high-x)
  k=4  SE corner  (low-y,  high-x)

Polygon vertex order used here: NW→SW→SE→NE  (k-indices [1, 2, 4, 3])

BOUT (ix, iy) ↔ gridue (i_pol, j_rad) mapping
  j_rad = ix - 1                   (valid for ix = 2 .. nx-3)
  i_pol = iy + 1  if iy < ny_inner
  i_pol = iy + 3  if iy >= ny_inner
The four outermost radial rows (ix=0,1 and ix=nx-2,nx-1) are extrapolated
guard cells not stored in rm/zm and are therefore skipped.

Usage
-----
    python visualize_in_grid.py <grid_file> [--field FIELD] [--cmap CMAP]
                                            [--no-edges] [--save FILE]
                                            [--plot_sep | --no-plot_sep]
                                            [--limiter PATH]

Examples
--------
    python visualize_in_grid.py IN_test1_bout_from_in.grd.nc
    python visualize_in_grid.py IN_test1_bout_from_in.grd.nc --field psixy
    python visualize_in_grid.py IN_test1_bout_from_in.grd.nc --field Bxy --cmap plasma --save Bxy.png
"""

import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import netCDF4


# Per-cut colour convention. Each branch cut gets its own colour; all dashed.
_BRANCH_CUT_COLORS = {
    "jyseps1_1": "gold",
    "jyseps2_1": "limegreen",
    "jyseps1_2": "cornflowerblue",
    "jyseps2_2": "tomato",
    "ny_inner": "cyan",
}


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


def build_cell_polygons(ds, topo):
    """
    Build a list of matplotlib Polygon patches, one per grid cell, using
    the rm/zm arrays stored in INGRID BOUT++ output files.

    rm/zm shape: (n_pol, n_rad, 5)
      k=0 centre, k=1 NW, k=2 SW, k=3 NE, k=4 SE

    Polygon vertex order (counter-clockwise in R-Z space): NW→SW→SE→NE

    BOUT index mapping to gridue array indices:
      j_rad = ix - 1
      i_pol = iy + 1   (if iy < ny_inner)
      i_pol = iy + 3   (if iy >= ny_inner)

    The four outermost radial rows (ix = 0, 1, nx-2, nx-1) are extrapolated
    guard cells not stored in rm/zm and are therefore skipped.

    Returns
    -------
    patches : list of mpatches.Polygon
    ix_valid : int  — number of valid radial cells (nx - 4)
    ny       : int  — number of poloidal cells
    """
    rm = ds.variables["rm"][:].data   # (n_pol, n_rad, 5)
    zm = ds.variables["zm"][:].data

    nx       = topo["nx"]
    ny       = topo["ny"]
    ny_inner = topo["ny_inner"]

    # Corner index order for a valid quadrilateral: NW→SW→SE→NE
    k = [1, 2, 4, 3]

    patches = []
    for ix in range(2, nx - 2):        # skip outermost 2 guard cells each side
        j_rad = ix - 1
        for iy in range(ny):
            i_pol = (iy + 1) if iy < ny_inner else (iy + 3)
            verts = np.column_stack([rm[i_pol, j_rad, k],
                                     zm[i_pol, j_rad, k]])
            patches.append(mpatches.Polygon(verts, closed=True))

    nx_valid = nx - 4
    return patches, nx_valid, ny


def build_branch_cut_polygons(ds, topo):
    """
    Build polygons for the two poloidal guard cells that sit at the inner
    target-plate branch cut (between BOUT iy=ny_inner-1 and iy=ny_inner).

    In the gridue these are at i_pol = ny_inner+1 and ny_inner+2; they are
    skipped by the standard BOUT index mapping but contain real geometry, so
    they must be plotted to avoid a visible gap in the grid.
    """
    rm = ds.variables["rm"][:].data
    zm = ds.variables["zm"][:].data

    nx       = topo["nx"]
    ny_inner = topo["ny_inner"]
    k = [1, 2, 4, 3]

    patches = []
    for ix in range(2, nx - 2):
        j_rad = ix - 1
        for i_pol_gc in [ny_inner + 1, ny_inner + 2]:
            verts = np.column_stack([rm[i_pol_gc, j_rad, k],
                                     zm[i_pol_gc, j_rad, k]])
            patches.append(mpatches.Polygon(verts, closed=True))
    return patches


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_grid_only(ds, topo, ax, edgecolor="black", linewidth=0.3):
    """Draw cell outlines with no fill."""
    print("Building cell polygons...")
    patches, nx_valid, ny = build_cell_polygons(ds, topo)
    gc_patches = build_branch_cut_polygons(ds, topo)
    col = PatchCollection(patches + gc_patches, facecolor="none",
                          edgecolor=edgecolor, linewidth=linewidth)
    ax.add_collection(col)
    print(f"  Drew {nx_valid * ny + len(gc_patches)} cell outlines "
          f"({len(gc_patches)} branch-cut guard cells included).")


def plot_field(ds, topo, field_name, ax, cmap="inferno",
               show_edges=True, edge_linewidth=0.15):
    """Fill cells with colour according to a 2D field, return the collection."""
    if field_name not in ds.variables:
        raise KeyError(
            f"Field '{field_name}' not found in the grid file.\n"
            f"Available 2D fields: {[k for k, v in ds.variables.items() if len(v.shape)==2]}"
        )

    nx       = topo["nx"]
    ny_inner = topo["ny_inner"]

    # Extract values only for the valid radial range ix=2..nx-3, all iy
    values_2d = ds.variables[field_name][:].data   # shape (nx, ny)
    values = values_2d[2:nx-2, :].flatten(order="C")

    print(f"Building cell polygons for field '{field_name}'...")
    patches, nx_valid, ny = build_cell_polygons(ds, topo)

    edgecol = "k" if show_edges else "none"
    col = PatchCollection(patches, cmap=cmap,
                          edgecolor=edgecol, linewidth=edge_linewidth)
    col.set_array(values)
    ax.add_collection(col)
    plt.colorbar(col, ax=ax, shrink=0.8, label=field_name)

    # Overlay branch-cut guard cells as unfilled outlines (no field value)
    gc_patches = build_branch_cut_polygons(ds, topo)
    gc_col = PatchCollection(gc_patches, facecolor="none",
                             edgecolor=edgecol, linewidth=edge_linewidth)
    ax.add_collection(gc_col)

    print(f"  Drew {nx_valid * ny} filled cells + {len(gc_patches)} branch-cut guard cells.")
    return col


def plot_branch_cuts(ax, rm, zm, topo, lw=1.5, linestyle="--"):
    """
    Overlay the topological branch cuts on the grid, drawn on the *cell edge*
    (not the cell centre).

    The gridue corner arrays are transposed to ``[radial, poloidal, corner]``
    and each cut is drawn along a fixed poloidal column. Corner index 4 (the
    radial-running cell face) is used for the ``jyseps`` cuts; corner index 3
    (the poloidal-facing / target face) for the ``ny_inner`` inner-target line.

    The poloidal column for a BOUT y-cut at index ``J`` accounts for the gridue
    guard cells: ``J + 1`` below ``ny_inner``, ``J + 3`` above it.

    Radial extent (SF topology, taken from the grid rather than hard-coded):
    ``jyseps1_1`` / ``jyseps2_1`` run out to ``ixseps1``; ``jyseps1_2`` /
    ``jyseps2_2`` run out to ``ixseps2``; ``ny_inner`` spans the full radial
    width on the target face.
    """
    ny_inner = topo["ny_inner"]
    r = rm.transpose(1, 0, 2)
    z = zm.transpose(1, 0, 2)
    n_rad, n_pol = r.shape[0], r.shape[1]

    # gridue poloidal column for a BOUT y-cut at index J (guard-cell offset).
    def ipol(J):
        return J + 1 if J < ny_inner else J + 3

    # Radial extent (separatrix index) each cut terminates at.
    ix1, ix2 = topo.get("ixseps1"), topo.get("ixseps2")
    cut_extent = {
        "jyseps1_1": ix1, "jyseps2_1": ix1,
        "jyseps1_2": ix2, "jyseps2_2": ix2,
    }

    drawn = False
    for name in ("jyseps1_1", "jyseps2_1", "jyseps1_2", "jyseps2_2"):
        J = topo.get(name)
        if J is None:
            continue
        ip = ipol(J)
        if not (0 <= ip < n_pol):
            continue
        ext = cut_extent[name]
        rad_max = n_rad if ext is None else min(ext, n_rad)
        ax.plot(r[0:rad_max, ip, 4], z[0:rad_max, ip, 4],
                color=_BRANCH_CUT_COLORS[name], lw=lw, ls=linestyle,
                label=f"{name}={J}", zorder=5)
        drawn = True

    # ny_inner: inner/outer divertor target split, full radial extent, target face.
    if ny_inner is not None and 0 <= ny_inner + 1 < n_pol:
        ax.plot(r[:, ny_inner + 1, 3], z[:, ny_inner + 1, 3],
                color=_BRANCH_CUT_COLORS["ny_inner"], lw=lw, ls=linestyle,
                label=f"ny_inner={ny_inner}", zorder=5)
        drawn = True

    return drawn


def plot_separatrices(ax, rm, zm, topo, lw=1.5):
    """
    Overlay the separatrices (``ixseps1`` magenta, ``ixseps2`` yellow) on the
    grid, on the cell edge (corner 1 of the cells at radial index ``ixseps-1``
    in the transposed corner array).

    ``ixseps1`` is the open SOL separatrix: it spans the whole poloidal domain,
    drawn in two arcs split at the ``ny_inner`` inner/outer target gap.

    ``ixseps2`` bounds the closed core, so it is drawn in pieces around the
    lower X-point, which reconnects the inner and outer sides differently:
      - core ring: the arc ``jyseps1_1+1 .. jyseps2_1`` closed on itself;
      - leg: the lower leg continues across the X-point on the outer side, i.e.
        ``jyseps1_1`` is joined to ``jyseps2_1+1``.
    The outer (above ``ny_inner``) half is drawn continuously.
    """
    ny_inner = topo["ny_inner"]
    r = rm.transpose(1, 0, 2)
    z = zm.transpose(1, 0, 2)
    n_rad, n_pol = r.shape[0], r.shape[1]
    corner = 1  # separatrix edge

    def ipol(J):
        return J + 1 if J < ny_inner else J + 3

    def plot_cols(k, cols, color, close=False, label=None):
        """Plot the separatrix (radial index k) along a list of poloidal columns."""
        cols = [c for c in cols if 0 <= c < n_pol]
        if len(cols) < 2:
            return False
        R = [r[k, c, corner] for c in cols]
        Z = [z[k, c, corner] for c in cols]
        if close:
            R.append(R[0])
            Z.append(Z[0])
        ax.plot(R, Z, color=color, lw=lw, label=label, zorder=6)
        return True

    drawn = False

    # -- ixseps1: open SOL separatrix, inner + outer arcs (split at ny_inner gap) --
    ix1 = topo.get("ixseps1")
    if ix1 is not None and 0 <= ix1 - 1 < n_rad:
        k = ix1 - 1
        plot_cols(k, range(0, ny_inner + 2), "magenta", label=f"ixseps1={ix1}")
        plot_cols(k, range(ny_inner + 3, n_pol), "magenta")
        drawn = True

    # -- ixseps2: closed core ring + leg joined across the lower X-point --
    ix2 = topo.get("ixseps2")
    if ix2 is not None and 0 <= ix2 - 1 < n_rad:
        k = ix2 - 1
        j11, j21 = topo.get("jyseps1_1"), topo.get("jyseps2_1")
        if j11 is not None and j21 is not None:
            # Core ring: closed arc between the X-points.
            plot_cols(k, range(ipol(j11 + 1), ipol(j21) + 1), "yellow",
                      close=True, label=f"ixseps2={ix2}")
            # Lower leg: below jyseps1_1, continuing from jyseps2_1+1 (the jump
            # between the two ranges draws the jyseps1_1 -> jyseps2_1+1 join).
            leg = list(range(0, ipol(j11) + 1)) + \
                list(range(ipol(j21 + 1), ny_inner + 2))
            plot_cols(k, leg, "yellow")
            # Outer half (above ny_inner), drawn continuously.
            plot_cols(k, range(ny_inner + 3, n_pol), "yellow")
        else:
            plot_cols(k, range(0, ny_inner + 2), "yellow", label=f"ixseps2={ix2}")
            plot_cols(k, range(ny_inner + 3, n_pol), "yellow")
        drawn = True

    return drawn


def plot_wall(ds, ax):
    """Draw the machine wall contour if present in the file."""
    if "closed_wall_R" not in ds.variables:
        return
    Rw = ds.variables["closed_wall_R"][:].data
    Zw = ds.variables["closed_wall_Z"][:].data
    ax.plot(Rw, Zw, color="dimgray", lw=2.0, label="wall", zorder=4)


def load_limiter(path):
    """
    Load a limiter/wall contour as (R, Z) arrays.

    Accepts a two-column text file of R Z pairs (whitespace- or comma-delimited,
    with optional ``#``/``%`` comment or header lines), or a netCDF file holding
    a recognised pair of R/Z variables. Returns (R, Z) 1D arrays in metres.
    """
    if path.endswith((".nc", ".cdf", ".netcdf")):
        lim = netCDF4.Dataset(path)
        candidates = [
            ("R", "Z"), ("r", "z"),
            ("limiter_R", "limiter_Z"),
            ("rlim", "zlim"), ("Rlim", "Zlim"),
            ("closed_wall_R", "closed_wall_Z"),
        ]
        for rname, zname in candidates:
            if rname in lim.variables and zname in lim.variables:
                R = lim.variables[rname][:].data.ravel()
                Z = lim.variables[zname][:].data.ravel()
                lim.close()
                return R, Z
        avail = list(lim.variables.keys())
        lim.close()
        raise KeyError(
            f"No recognised limiter R/Z variable pair in '{path}'. "
            f"Available variables: {avail}"
        )

    # Text file: let numpy figure out the delimiter, skipping comment lines.
    try:
        data = np.loadtxt(path, comments=("#", "%"))
    except ValueError:
        data = np.loadtxt(path, comments=("#", "%"), delimiter=",")
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(
            f"Limiter file '{path}' must have at least two columns (R, Z); "
            f"got shape {data.shape}."
        )
    return data[:, 0], data[:, 1]


def plot_limiter(path, ax, lw=2.0):
    """Overlay a limiter contour read from ``path``."""
    R, Z = load_limiter(path)
    ax.plot(R, Z, color="black", lw=lw, label="limiter", zorder=7)
    print(f"  Drew limiter contour from '{path}' ({len(R)} points).")


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
        description="Visualize an INGRID-generated BOUT++ grid file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("grid_file", help="Path to INGRID BOUT++ .nc grid file")
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
    parser.add_argument(
        "--plot_sep", action=argparse.BooleanOptionalAction, default=True,
        help="Overlay separatrices and branch cuts on the grid "
             "(use --no-plot_sep to disable; default: True)",
    )
    parser.add_argument(
        "--limiter", default=None, metavar="PATH",
        help="Overlay a limiter contour read from PATH (two-column R Z text "
             "file, or a netCDF file with an R/Z variable pair). "
             "If omitted, no limiter is drawn.",
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
        title = f"INGRID grid — {args.grid_file}"
    else:
        plot_field(ds, topo, args.field, ax,
                   cmap=args.cmap,
                   show_edges=not args.no_edges,
                   edge_linewidth=0.15)
        title = f"{args.field} — {args.grid_file}"

    if args.plot_sep:
        rm = ds.variables["rm"][:].data
        zm = ds.variables["zm"][:].data
        plot_branch_cuts(ax, rm, zm, topo)
        plot_separatrices(ax, rm, zm, topo)
    plot_wall(ds, ax)
    if args.limiter:
        plot_limiter(args.limiter, ax)
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
