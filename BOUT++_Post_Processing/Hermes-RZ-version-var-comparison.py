#!/usr/bin/env python3
"""Compare one variable between two BOUT++/Hermes-3 dump dirs and pinpoint where they diverge.

Like Hermes-version-var-comparison.py, but:
  * 2D plots use physical R, Z coordinates pulled from the supplied grid file
    (Rxy, Zxy). Falls back to index space if those aren't in the grid file.
  * 1D plots use the R coordinate along the slice (radial) or Z (poloidal).
  * Reports the first timestep at which the two runs disagree (within --rtol/--atol),
    and for 2D data also the first (x, y) cell where they disagree, with its R, Z.

The grid file (BOUT++ .nc grid, typically produced by hypnotoad) is required as
the first argument because dump files don't carry Rxy/Zxy themselves.

Usage:
    python Hermes-RZ-version-var-comparison.py grid.nc A_dir B_dir Ne
    python Hermes-RZ-version-var-comparison.py grid.nc A_dir B_dir Te --dim theta --idx 32
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from boutdata import collect
from boututils.datafile import DataFile


RADIAL = {"R", "x"}
POLOIDAL = {"theta", "y"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _list_vars(path):
    dmp = sorted(Path(path).glob("BOUT.dmp.*.nc"))
    if not dmp:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files in {path}")
    with DataFile(str(dmp[0])) as f:
        return set(f.list())


def collect_full(name, path):
    """Collect a variable as a numpy array (all timesteps if applicable)."""
    return np.asarray(collect(name, path=path, info=False, strict=True))


def collect_time_slice(name, path, tind):
    """Collect a variable at a single time index, then squeeze."""
    data = collect_full(name, path)
    if data.ndim >= 1 and data.shape[0] > 1 and "t_array" in _list_vars(path):
        try:
            data = data[tind]
        except IndexError:
            raise IndexError(
                f"--time {tind} out of range for {name} with shape {data.shape}"
            )
    return np.squeeze(data)


def get_grid_rz(grid_path):
    """Read Rxy, Zxy from a BOUT++ grid file. Returns (None, None) on failure."""
    p = Path(grid_path)
    if not p.is_file():
        print(f"WARNING: grid file {grid_path} not found — plotting in index space.")
        return None, None
    try:
        with DataFile(str(p)) as f:
            keys = set(f.list())
            if "Rxy" not in keys or "Zxy" not in keys:
                print(f"WARNING: {grid_path} has no Rxy/Zxy — plotting in index space.")
                return None, None
            R = np.squeeze(np.asarray(f.read("Rxy")))
            Z = np.squeeze(np.asarray(f.read("Zxy")))
    except Exception as exc:
        print(f"WARNING: failed to read grid {grid_path}: {exc}")
        return None, None
    return R, Z


def get_t_array(path):
    if "t_array" in _list_vars(path):
        return np.asarray(collect("t_array", path=path, info=False, strict=True))
    return None


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

def first_disagreement(a_full, b_full, t_array, rtol, atol):
    """Return the first time index at which a and b differ, or None.

    If the variable is 2D-in-space (a.shape after time is 2D), also returns
    the first (i, j) cell of disagreement at that timestep.
    """
    if a_full.shape != b_full.shape:
        return {"reason": f"shapes differ ({a_full.shape} vs {b_full.shape})"}

    has_time = (
        t_array is not None
        and a_full.ndim >= 1
        and a_full.shape[0] == t_array.size
    )
    if not has_time:
        # No time axis on this variable.
        mask = ~np.isclose(a_full, b_full, rtol=rtol, atol=atol, equal_nan=True)
        if not mask.any():
            return None
        first_cell = tuple(int(c[0]) for c in np.where(mask))
        return {"time_index": None, "time_value": None,
                "cell": first_cell, "static": True}

    n_t = a_full.shape[0]
    for ti in range(n_t):
        diff_mask = ~np.isclose(a_full[ti], b_full[ti],
                                rtol=rtol, atol=atol, equal_nan=True)
        if diff_mask.any():
            result = {"time_index": ti, "time_value": float(t_array[ti])}
            slice_shape = a_full[ti].shape
            if len(slice_shape) >= 2:
                idx = np.argwhere(diff_mask)
                # Pick the lexicographically-first (i, j[, ...]) cell.
                first = tuple(int(v) for v in idx[0])
                result["cell"] = first
            else:
                idx = np.where(diff_mask)[0]
                result["cell"] = (int(idx[0]),)
            return result
    return None


def describe_divergence(div, R=None, Z=None):
    if div is None:
        return "no disagreement within tolerance"
    if "reason" in div:
        return f"cannot compare: {div['reason']}"

    parts = []
    if div.get("static"):
        parts.append("static field (no time axis)")
    elif div.get("time_index") is not None:
        parts.append(
            f"first disagree at t-index {div['time_index']} "
            f"(t={div['time_value']:.6e})"
        )

    cell = div.get("cell")
    if cell is not None:
        if len(cell) == 2:
            i, j = cell
            parts.append(f"first cell (x={i}, y={j})")
            if R is not None and Z is not None and R.shape == Z.shape and \
               i < R.shape[0] and j < R.shape[1]:
                parts.append(f"R={R[i, j]:.4f} m, Z={Z[i, j]:.4f} m")
        elif len(cell) == 1:
            parts.append(f"first index {cell[0]}")
        else:
            parts.append(f"first cell {cell}")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_2d(name, a, b, R, Z, path_a, path_b, save, mark_cell=None):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"2D plot needs 2D data after squeezing; got shapes {a.shape} and {b.shape}. "
            "Pass --dim to make a 1D comparison."
        )

    use_rz = (
        R is not None and Z is not None and R.shape == a.shape and Z.shape == a.shape
    )
    if not use_rz:
        print("Note: Rxy/Zxy not available or shape mismatch — plotting in index space.")

    diff = a - b
    vmin = float(min(a.min(), b.min()))
    vmax = float(max(a.max(), b.max()))
    dlim = float(max(abs(diff.min()), abs(diff.max())))
    if dlim == 0.0:
        dlim = 1.0  # avoid degenerate colour scale on identical arrays

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)

    def _draw(ax, vals, vmin_, vmax_, cmap):
        if use_rz:
            return ax.pcolormesh(R, Z, vals, vmin=vmin_, vmax=vmax_,
                                 shading="auto", cmap=cmap)
        return ax.pcolormesh(vals.T, vmin=vmin_, vmax=vmax_,
                             shading="auto", cmap=cmap)

    im0 = _draw(axes[0], a, vmin, vmax, "viridis")
    axes[0].set_title(f"A: {Path(path_a).name}")
    fig.colorbar(im0, ax=axes[0])

    im1 = _draw(axes[1], b, vmin, vmax, "viridis")
    axes[1].set_title(f"B: {Path(path_b).name}")
    fig.colorbar(im1, ax=axes[1])

    im2 = _draw(axes[2], diff, -dlim, dlim, "RdBu_r")
    axes[2].set_title("A − B")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        if use_rz:
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            ax.set_aspect("equal")
        else:
            ax.set_xlabel("x (radial index)")
            ax.set_ylabel("y (poloidal index)")

    # Mark the first-disagreement cell (across all time) on the diff panel.
    if mark_cell is not None and len(mark_cell) == 2:
        i, j = mark_cell
        if use_rz and i < R.shape[0] and j < R.shape[1]:
            axes[2].plot(R[i, j], Z[i, j], marker="o", ms=10, mfc="none",
                         mec="lime", mew=2, label="first diff")
            axes[2].legend(loc="upper right")
        else:
            axes[2].plot(i, j, marker="o", ms=10, mfc="none",
                         mec="lime", mew=2, label="first diff")
            axes[2].legend(loc="upper right")

    fig.suptitle(f"{name}: 2D comparison")

    if save:
        out = Path.cwd() / f"compare_RZ_{name}_2d.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    plt.show()


def plot_1d(name, a, b, dim, idx, R, Z, path_a, path_b, save, mark_time=None,
            a_full=None, b_full=None, t_array=None):
    """1D line plot at the chosen slice; bottom subplot shows A−B vs time at the same cell.

    The mark_time arrow highlights the first timestep at which the two runs diverge
    on the chosen slice.
    """
    radial = dim in RADIAL
    a1 = _reduce_to_1d(a, radial, idx)
    b1 = _reduce_to_1d(b, radial, idx)

    if a1.shape != b1.shape:
        raise ValueError(
            f"1D shapes differ after slicing: {a1.shape} vs {b1.shape}"
        )

    # Build axis values from R or Z if we can.
    axis_vals = _axis_coords(a, R, Z, radial, idx)
    axis_label = ("R [m]" if radial else "Z [m]") if axis_vals is not None else \
                 ("x (radial index)" if radial else "y (poloidal index)")
    if axis_vals is None:
        axis_vals = np.arange(a1.size)

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 2]}, constrained_layout=True,
    )
    ax, ax_t = axes

    ax.plot(axis_vals, a1, label=f"A: {Path(path_a).name}", marker="o", ms=3)
    ax.plot(axis_vals, b1, label=f"B: {Path(path_b).name}", marker="x", ms=4, linestyle="--")
    ax.set_xlabel(axis_label)
    ax.set_ylabel(name)
    ax.set_title(f"{name} along {dim} (at final time)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Time-trace at the chosen slice: max|A-B| over the slice for each timestep.
    drew_time = False
    if a_full is not None and b_full is not None and t_array is not None \
            and a_full.shape == b_full.shape and a_full.ndim >= 2 \
            and a_full.shape[0] == t_array.size:
        try:
            a_slice_t = _slice_full_to_1d(a_full, radial, idx)
            b_slice_t = _slice_full_to_1d(b_full, radial, idx)
            err = np.max(np.abs(a_slice_t - b_slice_t), axis=1)
            ax_t.plot(t_array, err, color="firebrick")
            ax_t.set_yscale("symlog", linthresh=max(err.max() * 1e-6, 1e-30))
            ax_t.set_xlabel("time [s]")
            ax_t.set_ylabel("max |A − B| on slice")
            ax_t.grid(alpha=0.3)
            if mark_time is not None:
                ax_t.axvline(mark_time, color="lime", lw=1.5,
                             label=f"first diverge t={mark_time:.3e}")
                ax_t.legend(loc="best")
            drew_time = True
        except Exception as exc:
            print(f"(could not draw time-trace: {exc})")

    if not drew_time:
        # Fall back to plain residual at final time.
        ax_t.plot(axis_vals, a1 - b1, color="firebrick")
        ax_t.axhline(0, color="black", lw=0.5)
        ax_t.set_xlabel(axis_label)
        ax_t.set_ylabel("A − B (final time)")
        ax_t.grid(alpha=0.3)

    if save:
        out = Path.cwd() / f"compare_RZ_{name}_{dim}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    plt.show()


def _reduce_to_1d(data, radial, idx):
    if data.ndim == 1:
        return data
    if data.ndim != 2:
        raise ValueError(
            f"Cannot reduce shape {data.shape} to 1D; expected 1 or 2 dims after squeeze."
        )
    if radial:
        y_idx = idx if idx is not None else data.shape[1] // 2
        return data[:, y_idx]
    x_idx = idx if idx is not None else data.shape[0] // 2
    return data[x_idx, :]


def _slice_full_to_1d(data, radial, idx):
    """Apply the same slice as _reduce_to_1d but on (t, x, y) data; returns (t, N)."""
    if data.ndim == 2:
        return data  # already (t, N)
    if data.ndim != 3:
        raise ValueError(
            f"Expected (t, x, y) data for time-trace slicing; got shape {data.shape}"
        )
    if radial:
        y_idx = idx if idx is not None else data.shape[2] // 2
        return data[:, :, y_idx]
    x_idx = idx if idx is not None else data.shape[1] // 2
    return data[:, x_idx, :]


def _axis_coords(data, R, Z, radial, idx):
    """Return the physical coordinate along the chosen slice, if R/Z available."""
    if R is None or Z is None:
        return None
    if R.shape != data.shape:
        return None
    if radial:
        y_idx = idx if idx is not None else data.shape[1] // 2
        return R[:, y_idx]
    x_idx = idx if idx is not None else data.shape[0] // 2
    return Z[x_idx, :]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("grid", help="Path to the BOUT++ grid .nc file (provides Rxy, Zxy)")
    parser.add_argument("path_a", help="First simulation directory")
    parser.add_argument("path_b", help="Second simulation directory")
    parser.add_argument("variable", help="Variable name to compare (e.g. Ne, Te, Vd+)")
    parser.add_argument("--dim", choices=sorted(RADIAL | POLOIDAL), default=None,
                        help="If set, make a 1D plot vs this dimension. "
                             "R/x = radial, theta/y = poloidal.")
    parser.add_argument("--idx", type=int, default=None,
                        help="Index along the orthogonal axis for 1D plots "
                             "(defaults to the middle index).")
    parser.add_argument("--time", "-t", type=int, default=-1,
                        help="Time index to plot (default: -1, i.e. final).")
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative tolerance for divergence detection (default: 1e-5)")
    parser.add_argument("--atol", type=float, default=1e-8,
                        help="Absolute tolerance for divergence detection (default: 1e-8)")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save the figure as a PNG in the current directory.")
    args = parser.parse_args()

    print(f"Grid: {args.grid}")
    print(f"A: {args.path_a}")
    print(f"B: {args.path_b}")
    print(f"Variable: {args.variable}  |  plot time index: {args.time}")

    # --- load slice for plotting ---
    try:
        a_slice = collect_time_slice(args.variable, args.path_a, args.time)
        b_slice = collect_time_slice(args.variable, args.path_b, args.time)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Shape A: {a_slice.shape}  |  Shape B: {b_slice.shape}")
    if a_slice.shape != b_slice.shape:
        print("ERROR: shapes differ — cannot plot.", file=sys.stderr)
        sys.exit(1)

    # --- load full time series (only needed for divergence detection) ---
    try:
        a_full = collect_full(args.variable, args.path_a)
        b_full = collect_full(args.variable, args.path_b)
    except Exception as exc:
        print(f"WARNING: could not load full time series ({exc}); "
              "skipping divergence search.")
        a_full = b_full = None

    t_a = get_t_array(args.path_a)
    t_b = get_t_array(args.path_b)
    t_array = t_a if t_a is not None else t_b
    if t_a is not None and t_b is not None and t_a.shape == t_b.shape \
            and not np.allclose(t_a, t_b):
        print("WARNING: t_arrays differ between runs. Using A's t_array. "
              "Consider compare_dmp_diff_times.py if timesteps don't match.")

    # --- load grid R, Z ---
    R_grid, Z_grid = get_grid_rz(args.grid)

    # --- find divergence ---
    div_info = None
    if a_full is not None and b_full is not None:
        div_info = first_disagreement(a_full, b_full, t_array,
                                      args.rtol, args.atol)
        print(f"Divergence (rtol={args.rtol}, atol={args.atol}): "
              + describe_divergence(div_info, R_grid, Z_grid))

    # --- plot ---
    mark_cell = div_info.get("cell") if isinstance(div_info, dict) and \
                                        "cell" in div_info else None
    mark_time = div_info.get("time_value") if isinstance(div_info, dict) else None

    if args.dim is None:
        plot_2d(args.variable, a_slice, b_slice,
                R_grid, Z_grid, args.path_a, args.path_b, args.save,
                mark_cell=mark_cell)
    else:
        plot_1d(args.variable, a_slice, b_slice, args.dim, args.idx,
                R_grid, Z_grid, args.path_a, args.path_b, args.save,
                mark_time=mark_time,
                a_full=a_full, b_full=b_full, t_array=t_array)


if __name__ == "__main__":
    main()
