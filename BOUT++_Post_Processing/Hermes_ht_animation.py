#!/usr/bin/env python3
"""
Hermes_ht_animation.py — Animate a Hermes-3 simulation on a Hypnotoad grid.

This is the Hypnotoad-grid counterpart of Hermes_in_animation.py. Where that
script builds cell polygons from INGRID rm/zm corner arrays, this one uses the
Hypnotoad corner arrays (Rxy_corners, Rxy_lower_right_corners, ...) stored in
the grid file, then animates a simulation variable on top.

Polygons are built once from the corner arrays (tracking the BOUT (ix, iy)
index of each patch); each animation frame only updates the color array from
sim_data[ix, iy].

Usage
-----
    python Hermes_ht_animation.py <grid_file> <sim_results_dir> [options]

Example
-------
    python Hermes_ht_animation.py \\
        ht_try3.nc /path/to/sim \\
        --var Pe --zindex 0 --interval 100 --save Pe.mp4
"""

import argparse
import numpy as np
import xarray as xr
import xbout
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def build_polygons_ht(grid, MXG):
    """
    Build polygon patches from the Hypnotoad grid corner arrays, one per
    physical cell, tracking the BOUT (ix, iy) index of each patch so the
    matching simulation value can be looked up at animation time.

    Parameters
    ----------
    grid : xarray.Dataset
        Hypnotoad grid file loaded with xr.open_dataset.
    MXG : int
        Number of radial guard cells on each side to skip.

    Returns
    -------
    patches : list of matplotlib.patches.Polygon
    indices : list of (int, int)   -- BOUT (ix, iy) for each patch
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
            patches.append(mpatches.Polygon(np.array([R, Z]).T,
                                            fill=True, closed=True))
            indices.append((ix, iy))

    return patches, indices


def extract_colors(data_2d, indices):
    """Pull a color value for each polygon from a 2D (nx, ny) data slice."""
    return np.array([data_2d[ix, iy] for ix, iy in indices])


def physical_axis_limits(grid, MXG):
    """Return (R_lim, Z_lim) tuples from the physical-region corners only."""
    nx = int(grid["nx"].values)
    phys = slice(MXG, nx - MXG)
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
    return (R_phys.min(), R_phys.max()), (Z_phys.min(), Z_phys.max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate a Hermes-3 simulation on a Hypnotoad grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("grid_file", help="Path to Hypnotoad BOUT++ .nc grid file")
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
                        help="Fixed colour-scale maximum (default: global data max)")
    parser.add_argument("--fixed-clim", action="store_true",
                        help="(Deprecated; now the default) colour scale is always "
                             "held fixed across all frames")
    parser.add_argument("--no-edges", action="store_true",
                        help="Hide cell edge lines")
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

    # --- Load grid geometry ---
    grid = xr.open_dataset(args.grid_file, engine="netcdf4")
    nx_grid = int(grid["nx"].values)
    ny_grid = int(grid["ny"].values)
    print("Grid file:", args.grid_file)
    print(f"  nx={nx_grid}, ny={ny_grid}")

    # --- Load simulation results ---
    # No geometry attached: the corner arrays come from the grid file above,
    # and we only need the variable values indexed by BOUT (ix, iy).
    #
    # keep_yboundaries=True is REQUIRED: the Hypnotoad grid corner arrays
    # include the y-boundary guard cells (ny + 2*y_boundary_guards per
    # y-boundary region), so the sim must keep them too for the per-cell
    # (ix, iy) lookup in extract_colors to line up. Dropping them makes the
    # sim y-extent shorter than the polygon grid and the lookup runs off the
    # end of the array.
    print("Opening simulation dataset...")
    sim = xbout.open_boutdataset(
        args.sim_results + "/BOUT.dmp.*.nc",
        keep_yboundaries=True,
    )

    if args.var not in sim:
        raise KeyError(
            f"Variable '{args.var}' not found in the dataset.\n"
            f"Available evolving variables: "
            f"{[v for v in sim.data_vars if 't' in sim[v].dims]}"
        )

    MXG = int(sim.metadata["MXG"])

    da = sim[args.var]
    if "z" in da.dims:
        da = da.isel(z=args.zindex)
    # Expected dims after this: (t, x, y) -> (nt, nx, ny)
    all_data = da.transpose("t", "x", "y").values
    nt = all_data.shape[0]
    nx_sim, ny_sim = all_data.shape[1], all_data.shape[2]
    times = sim["t"].values
    print(f"  variable '{args.var}': {nt} time steps, (nx, ny) slice = ({nx_sim}, {ny_sim})")

    # --- Consistency check: sim array must index into the grid polygons ---
    # Polygons span BOUT ix = MXG..nx-MXG and iy = 0..ny_corners-1; we look up
    # sim_data[ix, iy] at those indices, so the sim array must be at least as
    # large as the grid CORNER arrays in both directions. Compare against the
    # corner-array shape (which includes y-boundary guard cells), not the
    # scalar grid ny, otherwise a y-boundary mismatch slips through and blows
    # up later inside extract_colors.
    nx_corners, ny_corners = grid["Rxy_corners"].shape
    if nx_sim < nx_corners - MXG or ny_sim < ny_corners:
        raise ValueError(
            "Simulation array does not match the grid topology.\n"
            f"  grid corner arrays:  nx={nx_corners}, ny={ny_corners} "
            f"(polygons need ix up to {nx_corners - MXG - 1}, iy up to {ny_corners - 1})\n"
            f"  sim :  nx={nx_sim}, ny={ny_sim}\n"
            "The grid file and the simulation were almost certainly produced "
            "with different resolutions / y-boundary settings, or the sim was "
            "loaded with keep_yboundaries=False. Check that this grid is the "
            "one the run actually used."
        )

    # --- Build polygons once, tracking BOUT (ix, iy) per patch ---
    patches, indices = build_polygons_ht(grid, MXG)
    print(f"  {len(patches)} filled cells.")

    # --- Colour scale ---
    # The colour scale is held CONSTANT across all frames (default: the global
    # min/max over the whole run) so the animation honestly shows the field
    # evolving. A per-frame rescaling would keep the colourbar shifting and
    # hide the actual change. Override the fixed range with --vmin / --vmax.
    global_min = np.nanmin(all_data)
    global_max = np.nanmax(all_data)
    vmin = args.vmin if args.vmin is not None else global_min
    vmax = args.vmax if args.vmax is not None else global_max

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 10))

    edgecol = "none" if args.no_edges else "face"
    collection = PatchCollection(
        patches, cmap=args.cmap, edgecolors=edgecol,
        linewidths=0.1, joinstyle="bevel",
    )
    colors0 = extract_colors(all_data[0], indices)
    collection.set_array(colors0)
    collection.set_clim(vmin=vmin, vmax=vmax)   # fixed for the whole animation
    ax.add_collection(collection)

    cb = plt.colorbar(collection, ax=ax, label=args.var, shrink=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")

    (rmin, rmax), (zmin, zmax) = physical_axis_limits(grid, MXG)
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(zmin, zmax)

    title = ax.set_title(f"{args.var}   t = {times[0]:.3f}")
    plt.tight_layout()

    def update(frame):
        # Only the colours change between frames; the clim stays fixed.
        colors = extract_colors(all_data[frame], indices)
        collection.set_array(colors)
        title.set_text(f"{args.var}   t = {times[frame]:.3f}")
        return collection, title

    if args.save:
        # Saving uses FuncAnimation, which iterates frames explicitly and
        # works reliably under the Agg writer path.
        anim = animation.FuncAnimation(
            fig, update, frames=nt, interval=args.interval, blit=False,
        )
        print(f"Saving animation to {args.save} ({nt} frames)...")
        if args.save.endswith(".gif"):
            anim.save(args.save, writer="pillow", fps=args.fps)
        else:
            anim.save(args.save, writer="ffmpeg", fps=args.fps)
        print("Done.")
    else:
        # Live display: drive frames manually with plt.pause instead of
        # FuncAnimation + plt.show(). On the macOS "macosx" backend the
        # FuncAnimation timer never starts under plt.show() and the window
        # freezes on the first frame. A manual loop steps reliably, but ONLY
        # if each frame forces a hard repaint: draw_idle() alone does not
        # repaint the macosx canvas from a plain Python loop, whereas
        # draw() + flush_events() does. Loops until the window is closed.
        print(f"Displaying animation on backend '{matplotlib.get_backend()}' "
              f"({nt} frames; close the window to stop)...")
        plt.show(block=False)
        try:
            while plt.fignum_exists(fig.number):
                for frame in range(nt):
                    if not plt.fignum_exists(fig.number):
                        break
                    update(frame)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    print(f"\r  frame {frame+1}/{nt}", end="", flush=True)
                    plt.pause(args.interval / 1000)
            print()
        except Exception:
            # Surface the real error instead of hiding it.
            print()
            import traceback
            traceback.print_exc()

    grid.close()


if __name__ == "__main__":
    main()
