#!/usr/bin/env python3
"""
Hermes_in_animation.py — Animate a Hermes-3 simulation on an INGRID grid.

This is the INGRID-grid counterpart of BOUT_ht_animation.py. Where that script
uses Hypnotoad corner arrays (Rxy_corners, ...), this one builds the cell
polygons from the INGRID rm/zm corner arrays — the same geometry used by
visualize_in_grid.py — and then animates a simulation variable on top.

INGRID BOUT++ grid files store cell-corner geometry as:
  rm   shape (n_pol, n_rad, 5)   R-coordinates
  zm   shape (n_pol, n_rad, 5)   Z-coordinates
with corner index order  k=0 centre, k=1 NW, k=2 SW, k=3 NE, k=4 SE
(polygon vertex order NW->SW->SE->NE, i.e. k = [1, 2, 4, 3]).

BOUT (ix, iy) <-> gridue (i_pol, j_rad) mapping:
  j_rad = ix - 1                      (valid for ix = 2 .. nx-3)
  i_pol = iy + 1   if iy <  ny_inner
  i_pol = iy + 3   if iy >= ny_inner
The four outermost radial rows (ix = 0, 1, nx-2, nx-1) are extrapolated guard
cells not stored in rm/zm and are therefore skipped.

The polygons are built once; each animation frame only updates the color array
from sim_data[ix, iy] at the BOUT (ix, iy) indices tracked per patch.

Usage
-----
    python Hermes_in_animation.py <grid_file> <sim_results_dir> [options]

Example
-------
    python Hermes_in_animation.py \\
        IN_test1_bout_from_in.grd.nc /path/to/sim \\
        --var Pe --zindex 0 --interval 100 --save Pe.mp4
"""

import argparse
import numpy as np
import xbout
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import netCDF4


# ---------------------------------------------------------------------------
# Grid / topology helpers (shared conventions with visualize_in_grid.py)
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
    Build polygon patches from the INGRID rm/zm corner arrays, one per physical
    cell, tracking the BOUT (ix, iy) index of each patch so the matching
    simulation value can be looked up at animation time.

    Returns
    -------
    patches : list of mpatches.Polygon
    indices : list of (int, int)   -- BOUT (ix, iy) for each patch
    nx_valid : int                 -- number of valid radial cells (nx - 4)
    ny       : int                 -- number of poloidal cells
    """
    rm = ds.variables["rm"][:].data   # (n_pol, n_rad, 5)
    zm = ds.variables["zm"][:].data

    nx       = topo["nx"]
    ny       = topo["ny"]
    ny_inner = topo["ny_inner"]

    # Corner index order for a valid quadrilateral: NW->SW->SE->NE
    k = [1, 2, 4, 3]

    patches = []
    indices = []
    for ix in range(2, nx - 2):        # skip outermost 2 guard cells each side
        j_rad = ix - 1
        for iy in range(ny):
            i_pol = (iy + 1) if iy < ny_inner else (iy + 3)
            verts = np.column_stack([rm[i_pol, j_rad, k],
                                     zm[i_pol, j_rad, k]])
            patches.append(mpatches.Polygon(verts, closed=True))
            indices.append((ix, iy))

    nx_valid = nx - 4
    return patches, indices, nx_valid, ny


def build_branch_cut_polygons(ds, topo):
    """
    Build polygons for the two poloidal guard cells at the inner target-plate
    branch cut (i_pol = ny_inner+1, ny_inner+2). These hold real geometry but
    have no simulation value, so they are drawn as static outlines only.
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


def extract_colors(data_2d, indices):
    """Pull a color value for each polygon from a 2D (nx, ny) data slice."""
    return np.array([data_2d[ix, iy] for ix, iy in indices])


def plot_separatrix_and_cuts(ds, topo, ax):
    """Overlay separatrices and branch cuts using cell-centre coordinates."""
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

    ix1c = min(ix1, nx - 1)
    ix2c = min(ix2, nx - 1)

    ax.plot(R[ix1c, :], Z[ix1c, :],
            color="magenta", lw=1.5, label=f"ixseps1={ix1}", zorder=5)
    if ix2c != ix1c and ix2 < nx:
        ax.plot(R[ix2c, :], Z[ix2c, :],
                color="yellow", lw=1.5, label=f"ixseps2={ix2}", zorder=5)

    ax.plot(R[:, jy11], Z[:, jy11],
            color="gold", lw=1.2, ls="--", label=f"jyseps1_1={jy11}", zorder=5)
    ax.plot(R[:, jy21], Z[:, jy21],
            color="limegreen", lw=1.2, ls="--", label=f"jyseps2_1={jy21}", zorder=5)
    ax.plot(R[:, ny_i], Z[:, ny_i],
            color="cyan", lw=1.2, ls="--", label=f"ny_inner={ny_i}", zorder=5)
    ax.plot(R[:, jy12], Z[:, jy12],
            color="cornflowerblue", lw=1.2, ls="--", label=f"jyseps1_2={jy12}", zorder=5)
    ax.plot(R[:, jy22], Z[:, jy22],
            color="tomato", lw=1.2, ls="--", label=f"jyseps2_2={jy22}", zorder=5)


def plot_wall(ds, ax):
    """Draw the machine wall contour if present in the file."""
    if "closed_wall_R" not in ds.variables:
        return
    Rw = ds.variables["closed_wall_R"][:].data
    Zw = ds.variables["closed_wall_Z"][:].data
    ax.plot(Rw, Zw, color="dimgray", lw=2.0, label="wall", zorder=6)


def set_axis_style(ax, ds):
    """Set axis limits, labels, aspect ratio from cell-centre coords."""
    R = ds.variables["Rxy"][:].data
    Z = ds.variables["Zxy"][:].data
    margin_R = 0.05 * (R.max() - R.min())
    margin_Z = 0.05 * (Z.max() - Z.min())
    ax.set_xlim(R.min() - margin_R, R.max() + margin_R)
    ax.set_ylim(Z.min() - margin_Z, Z.max() + margin_Z)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate a Hermes-3 simulation on an INGRID grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("grid_file", help="Path to INGRID BOUT++ .nc grid file")
    parser.add_argument("sim_results", help="Path to simulation results directory")
    parser.add_argument("--var", default="Pe",
                        help="Variable to animate (default: Pe)")
    parser.add_argument("--zindex", type=int, default=0,
                        help="Toroidal index, used only if the variable has a z dim (default: 0)")
    parser.add_argument("--cmap", default="Spectral_r",
                        help="Matplotlib colormap (default: Spectral_r)")
    parser.add_argument("--interval", type=int, default=100,
                        help="Delay between frames in ms (default: 100)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Fixed colour-scale minimum (default: global data min)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fixed colour-scale maximum (default: per-frame max)")
    parser.add_argument("--fixed-clim", action="store_true",
                        help="Hold both vmin and vmax fixed at the global data range")
    parser.add_argument("--no-edges", action="store_true",
                        help="Hide cell edge lines")
    parser.add_argument("--plot_sep", action=argparse.BooleanOptionalAction, default=False,
                        help="Overlay separatrices and branch cuts (default: off)")
    parser.add_argument("--save", default=None, metavar="FILE",
                        help="Save animation to FILE (.mp4 or .gif) instead of displaying")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for saved animation (default: 10)")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    # --- Load grid geometry and topology ---
    ds   = load_grid(args.grid_file)
    topo = get_topology_indices(ds)
    print("Grid file:", args.grid_file)
    print(f"  nx={topo['nx']}, ny={topo['ny']}, ny_inner={topo['ny_inner']}")

    # --- Load simulation results ---
    print("Opening simulation dataset...")
    sim = xbout.open_boutdataset(
        args.sim_results + "/BOUT.dmp.*.nc",
        gridfilepath=args.grid_file,
        geometry="toroidal",
        keep_yboundaries=False,
    )

    if args.var not in sim:
        raise KeyError(
            f"Variable '{args.var}' not found in the dataset.\n"
            f"Available evolving variables: "
            f"{[v for v in sim.data_vars if 't' in sim[v].dims]}"
        )

    da = sim[args.var]
    if "z" in da.dims:
        da = da.isel(z=args.zindex)
    # Expected dims after this: (t, x, y) -> (nt, nx, ny)
    all_data = da.transpose("t", "x", "y").values
    nt = all_data.shape[0]
    times = sim["t"].values
    print(f"  variable '{args.var}': {nt} time steps, slice shape {all_data.shape[1:]}")

    # --- Build polygons once, tracking BOUT (ix, iy) per patch ---
    patches, indices, nx_valid, ny = build_cell_polygons(ds, topo)
    gc_patches = build_branch_cut_polygons(ds, topo)
    print(f"  {nx_valid * ny} filled cells + {len(gc_patches)} branch-cut outlines.")

    # --- Colour scale ---
    global_min = np.nanmin(all_data)
    global_max = np.nanmax(all_data)
    vmin = args.vmin if args.vmin is not None else global_min
    if args.fixed_clim:
        vmax_fixed = args.vmax if args.vmax is not None else global_max
    else:
        vmax_fixed = args.vmax   # None => dynamic per-frame max

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 10))

    edgecol = "none" if args.no_edges else "k"
    collection = PatchCollection(
        patches, cmap=args.cmap, edgecolors=edgecol,
        linewidths=0.1, joinstyle="bevel",
    )
    colors0 = extract_colors(all_data[0], indices)
    collection.set_array(colors0)
    collection.set_clim(
        vmin=vmin,
        vmax=vmax_fixed if vmax_fixed is not None else np.nanmax(colors0),
    )
    ax.add_collection(collection)

    # Static branch-cut guard cells as outlines only
    gc_col = PatchCollection(gc_patches, facecolor="none",
                             edgecolor=edgecol, linewidth=0.1)
    ax.add_collection(gc_col)

    cb = plt.colorbar(collection, ax=ax, label=args.var, shrink=0.8)

    if args.plot_sep:
        plot_separatrix_and_cuts(ds, topo, ax)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    plot_wall(ds, ax)
    set_axis_style(ax, ds)

    title = ax.set_title(f"{args.var}   t = {times[0]:.3f}")
    plt.tight_layout()

    def update(frame):
        colors = extract_colors(all_data[frame], indices)
        collection.set_array(colors)
        if vmax_fixed is None:
            collection.set_clim(vmin=vmin, vmax=np.nanmax(colors))
        title.set_text(f"{args.var}   t = {times[frame]:.3f}")
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

    ds.close()


if __name__ == "__main__":
    main()
