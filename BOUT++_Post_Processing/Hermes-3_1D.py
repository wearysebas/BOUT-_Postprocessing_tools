#!/usr/bin/env python3
"""
1D profile analysis of a Hermes-3 / BOUT++ variable.

Take a single Hermes-3 run and plot one variable along a single coordinate
direction — either radially (vs major radius ``R``) or poloidally (vs the
poloidal angle ``theta``). The orthogonal coordinate is held fixed at one index.

This is the standalone, single-simulation descendant of
``Hermes-RTheta-version-var-comparison.py``: that script's ``--dim`` add-on did
exactly this R/theta slicing, but wrapped in two-run comparison + divergence
detection machinery we don't need here. The plotting / time behaviour mirrors
``Hermes-3_postprocessing.py`` (same ``--animate`` / ``--time`` semantics).

Coordinate conventions (BOUT++ layout, data shape ``(t, x, y, z)``):
    * Radial   = BOUT x. Plotting ``--R`` fixes a poloidal index ``y`` and runs
      along x; the abscissa is ``Rxy[:, y]`` (major radius along that line).
    * Poloidal = BOUT y. Plotting ``--theta`` fixes a radial index ``x`` and
      runs along y; the abscissa is ``theta``, built as the cumulative sum of
      ``dy`` so it is monotonic (unlike Z, which loops around a flux surface).

The grid file (BOUT++ ``.nc``, typically from hypnotoad) is required because the
dump files don't carry ``Rxy`` / ``dy``.

Why ``boutdata.collect`` and not the aligned xbout loader: ``collect`` returns
``(t, nx, ny, nz)`` that aligns cell-for-cell with the grid's ``Rxy`` / ``dy``,
so the abscissa is guaranteed correct.

Ragged dumps: an HPC run that's killed (or wraps up) rarely flushes the same
number of output steps on every MPI rank, so the per-rank ``BOUT.dmp.<n>.nc``
files disagree on the length of the time axis — which makes a naive read choke.
We scan every dump file for the *shortest* common time length and cap the read
with ``tind=[0, min_t-1]``, so ``collect`` never reaches past a step that some
rank is missing. Same idea as ``open_aligned_boutdataset`` in
``Hermes-3_postprocessing.py``, just expressed through ``collect``'s ``tind``.

Usage:
    python Hermes-3_1D.py <grid_file> <sim_results_dir> --var Te [options]

Examples:
    # Radial Te profile at the outboard midplane, final timestep:
    python Hermes-3_1D.py grid.nc run_dir --var Te --R

    # Poloidal Ne profile along the separatrix, animated over time:
    python Hermes-3_1D.py grid.nc run_dir --var Ne --theta --animate

    # Radial profile at a specific poloidal index and a specific time slice:
    python Hermes-3_1D.py grid.nc run_dir --var Pe --R --idx 32 --time 10
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from boutdata import collect
from boututils.datafile import DataFile


# ── Grid / data loading ──────────────────────────────────────────────────────

def get_grid_coords(grid_path):
    """
    Read the radial coordinate ``Rxy`` and build a 1D poloidal angle ``theta``
    from a BOUT++ grid file, along with the separatrix index for sensible
    default slice locations.

    ``theta`` is the BOUT++ poloidal (y) coordinate, built as
    ``theta[j] = sum_{k<j} dy[k]`` so the abscissa is monotonic. ``dy`` usually
    doesn't vary across x, so a 2D ``dy`` is collapsed along x with the median.

    Returns
    -------
    R : np.ndarray, shape (nx, ny)        major radius of every cell centre [m]
    theta : np.ndarray, shape (ny,)       cumulative poloidal angle [rad]
    ixseps1 : int or None                 primary separatrix radial index
    """
    with DataFile(str(grid_path)) as f:
        keys = set(f.list())
        if "Rxy" not in keys:
            raise KeyError(f"{grid_path} has no 'Rxy' — cannot build R/theta axes.")
        R = np.squeeze(np.asarray(f.read("Rxy")))

        # theta from cumsum(dy); fall back to the y-index if dy isn't usable.
        theta = None
        if "dy" in keys:
            dy = np.squeeze(np.asarray(f.read("dy")))
            if dy.ndim == 2:
                dy_1d = np.median(dy, axis=0)   # dy ~ constant across x
            elif dy.ndim == 1:
                dy_1d = dy
            else:
                dy_1d = None
            if dy_1d is not None and dy_1d.size == R.shape[1]:
                theta = np.concatenate(([0.0], np.cumsum(dy_1d)))[:R.shape[1]]
                print(f"[Hermes-3_1D] theta from cumsum(dy): "
                      f"[{theta.min():.4f}, {theta.max():.4f}] rad")
        if theta is None:
            theta = np.arange(R.shape[1], dtype=float)
            print("[Hermes-3_1D] dy not usable — using y-index as theta.")

        ixseps1 = int(np.squeeze(f.read("ixseps1"))) if "ixseps1" in keys else None

    return R, theta, ixseps1


def _proc_index(path):
    """Sort key: numeric processor index from a BOUT.dmp.<n>.nc filename."""
    m = re.search(r"BOUT\.dmp\.(\d+)\.nc$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def common_tlen(path):
    """
    Shortest time-axis length across the ``BOUT.dmp.*.nc`` files in ``path``.

    When a run is killed mid-output the ranks don't all flush the same number
    of steps, so we read each file's ``t_array`` length and return the minimum.
    Capping ``collect`` at this many steps avoids reading past a timestep that
    some rank never wrote. Returns ``None`` if no time axis is present (static
    grid-only data) or the dump files can't be inspected.
    """
    files = sorted(glob.glob(os.path.join(path, "BOUT.dmp.*.nc")), key=_proc_index)
    if not files:
        return None

    tlens = []
    for fp in files:
        try:
            with DataFile(fp) as f:
                if "t_array" in f.list():
                    tlens.append(int(np.atleast_1d(f.read("t_array")).size))
        except Exception:
            continue
    if not tlens:
        return None

    min_t, max_t = min(tlens), max(tlens)
    if min_t != max_t:
        n_short = sum(1 for n in tlens if n < max_t)
        print(f"[Hermes-3_1D] Ragged dumps: time lengths {min_t}..{max_t} "
              f"({n_short}/{len(tlens)} ranks short). Truncating to the last "
              f"common step (t index 0..{min_t - 1}).")
    else:
        print(f"[Hermes-3_1D] All {len(tlens)} dump files agree on {min_t} steps.")
    return min_t


def collect_var(var, path, zindex):
    """
    Collect ``var`` over the run (truncated to the last common step) and reduce
    it to ``(nt, nx, ny)``.

    Hermes-3 evolving fields come back as ``(t, x, y, z)``; the toroidal slice
    ``zindex`` is selected. A static field (no time axis) is given a length-1
    time axis so the caller can treat everything uniformly.

    Returns
    -------
    data : np.ndarray, shape (nt, nx, ny)
    t_array : np.ndarray, shape (nt,)   physical times (frame index if absent)
    """
    min_t = common_tlen(path)
    tind = [0, min_t - 1] if min_t is not None else None

    # Evolving fields accept tind; a static field has no time dim and would
    # reject it, so fall back to a plain read in that case.
    try:
        data = np.asarray(collect(var, path=path, info=False, strict=True,
                                  **({"tind": tind} if tind is not None else {})))
    except Exception:
        data = np.asarray(collect(var, path=path, info=False, strict=True))

    # Physical time axis (also truncated to the common range, if it exists).
    try:
        t_array = np.asarray(collect("t_array", path=path, info=False, strict=True))
        if min_t is not None:
            t_array = t_array[:min_t]
    except Exception:
        t_array = None
    nt = t_array.size if t_array is not None else None

    if data.ndim == 4:                       # (t, x, y, z) — evolving field
        data = data[:, :, :, zindex]
    elif data.ndim == 3:
        if nt is not None and data.shape[0] == nt:
            pass                             # (t, x, y) already
        else:                                # (x, y, z) static field
            data = data[:, :, zindex][None, ...]
    elif data.ndim == 2:                     # (x, y) static field
        data = data[None, ...]
    else:
        raise ValueError(f"{var!r} has unexpected shape {data.shape} after collect.")

    if t_array is None or t_array.size != data.shape[0]:
        t_array = np.arange(data.shape[0], dtype=float)

    return data, t_array


# ── Slicing ──────────────────────────────────────────────────────────────────

def default_index(dim, R, ixseps1):
    """
    Pick a physically meaningful default for the fixed (orthogonal) index.

    --R  (radial profile): fix the poloidal index at the outboard midplane,
         i.e. the y where R is largest on the separatrix flux surface.
    --theta (poloidal profile): fix the radial index at the separatrix.

    Falls back to the middle index when ``ixseps1`` isn't in the grid.
    """
    nx, ny = R.shape
    if dim == "R":
        if ixseps1 is not None and 0 <= ixseps1 < nx:
            y_mid = int(np.argmax(R[ixseps1, :]))
            print(f"[Hermes-3_1D] default poloidal slice = outboard midplane "
                  f"(y={y_mid}, from argmax Rxy[ixseps1]).")
            return y_mid
        print(f"[Hermes-3_1D] no ixseps1 — default poloidal slice = middle (y={ny // 2}).")
        return ny // 2
    else:  # theta
        if ixseps1 is not None and 0 <= ixseps1 < nx:
            print(f"[Hermes-3_1D] default radial slice = separatrix (x=ixseps1={ixseps1}).")
            return ixseps1
        print(f"[Hermes-3_1D] no ixseps1 — default radial slice = middle (x={nx // 2}).")
        return nx // 2


def slice_1d(frame, dim, idx):
    """Reduce a single ``(nx, ny)`` frame to a 1D profile along ``dim``."""
    if dim == "R":          # radial profile: vary x at fixed poloidal index
        return frame[:, idx]
    return frame[idx, :]    # poloidal profile: vary y at fixed radial index


def axis_coords(dim, idx, R, theta):
    """Abscissa values and label for the chosen 1D direction."""
    if dim == "R":
        return R[:, idx], "R [m]"
    return theta, r"$\theta$ [rad]"


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_static(var, data, t_array, dim, idx, R, theta, time, save):
    """Plot a single time slice as a 1D profile."""
    frame = data[time]
    y = slice_1d(frame, dim, idx)
    x, xlabel = axis_coords(dim, idx, R, theta)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", ms=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(var)
    other = "y" if dim == "R" else "x"
    ax.set_title(f"{var} vs {dim}  (fixed {other}={idx}, t={time} → "
                 f"{t_array[time]:.4g})")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"[Hermes-3_1D] Saved {save}")
    plt.show()


def animate_1d(var, data, t_array, dim, idx, R, theta, interval, save):
    """
    Animate the 1D profile over every timestep.

    The abscissa and y-limits are fixed once (limits span the whole time series
    so the axes are stable across frames); only the line data and title update.
    """
    from matplotlib.animation import FuncAnimation

    nt = data.shape[0]
    x, xlabel = axis_coords(dim, idx, R, theta)

    # Stable y-limits across the whole series (slice every frame first).
    sliced = np.stack([slice_1d(data[k], dim, idx) for k in range(nt)])
    ymin, ymax = np.nanmin(sliced), np.nanmax(sliced)
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0

    fig, ax = plt.subplots(figsize=(8, 5))
    (line,) = ax.plot(x, sliced[0], marker="o", ms=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(var)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(alpha=0.3)
    other = "y" if dim == "R" else "x"
    title = ax.set_title("")

    def update(frame):
        line.set_ydata(sliced[frame])
        title.set_text(f"{var} vs {dim}  (fixed {other}={idx}, "
                       f"t={t_array[frame]:.4g})")
        return line, title

    anim = FuncAnimation(fig, update, frames=nt, interval=interval,
                         blit=False, repeat=True)
    plt.tight_layout()

    if save:
        print(f"[Hermes-3_1D] Saving animation to {save} ...")
        anim.save(save)
        print("[Hermes-3_1D] Done.")
    else:
        plt.show()
    return anim


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="1D radial (R) or poloidal (theta) profile of a Hermes-3 variable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("grid_file", help="Path to the BOUT++ grid file (.nc); provides Rxy/dy")
    parser.add_argument("sim_results", help="Path to the simulation results directory")
    parser.add_argument("--var", required=True, help="Variable to plot (e.g. Te, Ne, Pe)")

    coord = parser.add_mutually_exclusive_group()
    coord.add_argument("--R", dest="dim", action="store_const", const="R",
                       help="Plot a radial profile vs major radius R (default)")
    coord.add_argument("--theta", dest="dim", action="store_const", const="theta",
                       help="Plot a poloidal profile vs poloidal angle theta")
    parser.set_defaults(dim="R")

    parser.add_argument("--idx", type=int, default=None,
                        help="Index along the fixed (orthogonal) axis: poloidal y "
                             "for --R, radial x for --theta. Default: outboard "
                             "midplane (--R) / separatrix (--theta).")
    parser.add_argument("--time", type=int, default=-1,
                        help="Time index for a static plot (default: -1 = last)")
    parser.add_argument("--zindex", type=int, default=0,
                        help="Toroidal index (default: 0)")
    parser.add_argument("--animate", action="store_true", default=False,
                        help="Animate over all timesteps instead of one slice")
    parser.add_argument("--interval", type=int, default=200,
                        help="Delay between animation frames in ms (default: 200)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save the figure/animation to this path instead of "
                             "showing it (e.g. profile.png, anim.gif / .mp4)")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    # Coordinates from the grid, variable from the dumps.
    R, theta, ixseps1 = get_grid_coords(args.grid_file)
    data, t_array = collect_var(args.var, args.sim_results, args.zindex)

    nx, ny = R.shape
    if data.shape[1:] != (nx, ny):
        print(f"WARNING: {args.var} slice shape {data.shape[1:]} != grid (nx, ny) "
              f"{(nx, ny)}. R/theta axes may be misaligned (guard cells?).",
              file=sys.stderr)

    # Resolve the fixed index (smart default unless the user pinned it).
    idx = args.idx if args.idx is not None else default_index(args.dim, R, ixseps1)
    limit = nx if args.dim == "theta" else ny
    if not (0 <= idx < limit):
        print(f"ERROR: --idx {idx} out of range [0, {limit}).", file=sys.stderr)
        sys.exit(1)

    if args.animate:
        animate_1d(args.var, data, t_array, args.dim, idx, R, theta,
                   args.interval, args.save)
    else:
        plot_static(args.var, data, t_array, args.dim, idx, R, theta,
                    args.time, args.save)


if __name__ == "__main__":
    main()
