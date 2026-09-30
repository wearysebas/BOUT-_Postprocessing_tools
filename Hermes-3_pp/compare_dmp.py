#!/usr/bin/env python3
"""Compare all quantities between two BOUT++/Hermes-3 simulation output directories.

Usage:
    python compare_dmp.py /path/to/sim_A /path/to/sim_B [--rtol 1e-5] [--atol 1e-8]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from boutdata import collect
from boututils.datafile import DataFile


def list_variables(path):
    """Return the sorted list of variable names in a BOUT++ dump directory."""
    dmp_files = sorted(Path(path).glob("BOUT.dmp.*.nc"))
    if not dmp_files:
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files found in {path}")
    with DataFile(str(dmp_files[0])) as f:
        return sorted(f.list())


def compare_variable(name, path_a, path_b, rtol, atol):
    """Collect `name` from both runs and compare element-wise."""
    try:
        a = collect(name, path=path_a, info=False, strict=True)
        b = collect(name, path=path_b, info=False, strict=True)
    except Exception as exc:
        return "ERROR", f"collect failed: {exc}"

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        return "SHAPE", f"shape mismatch: {a.shape} vs {b.shape}"

    if a.dtype.kind not in "fiub" or b.dtype.kind not in "fiub":
        equal = np.array_equal(a, b)
        return ("OK", "identical") if equal else ("FAIL", "non-numeric mismatch")

    mask = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    n_bad = int(mask.sum())
    if n_bad == 0:
        max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
        return "OK", f"max|Δ|={max_abs:.3e}"

    diff = np.abs(a - b)
    max_abs = float(diff.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = diff / np.maximum(np.abs(a), np.abs(b))
        rel = np.where(np.isfinite(rel), rel, 0.0)
    max_rel = float(rel.max())
    frac = n_bad / a.size
    return "FAIL", (
        f"{n_bad}/{a.size} ({frac:.2%}) points differ, "
        f"max|Δ|={max_abs:.3e}, max rel={max_rel:.3e}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path_a", help="First simulation directory")
    parser.add_argument("path_b", help="Second simulation directory")
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative tolerance (default: 1e-5)")
    parser.add_argument("--atol", type=float, default=1e-8,
                        help="Absolute tolerance (default: 1e-8)")
    args = parser.parse_args()

    vars_a = set(list_variables(args.path_a))
    vars_b = set(list_variables(args.path_b))

    only_a = sorted(vars_a - vars_b)
    only_b = sorted(vars_b - vars_a)
    common = sorted(vars_a & vars_b)

    print(f"A: {args.path_a}")
    print(f"B: {args.path_b}")
    print(f"rtol={args.rtol}, atol={args.atol}")
    print(f"{len(common)} common variables, "
          f"{len(only_a)} only in A, {len(only_b)} only in B\n")

    if only_a:
        print("Only in A:", ", ".join(only_a))
    if only_b:
        print("Only in B:", ", ".join(only_b))
    if only_a or only_b:
        print()

    n_ok = n_fail = n_err = 0
    failures = []
    width = max((len(v) for v in common), default=4)

    for name in common:
        status, detail = compare_variable(name, args.path_a, args.path_b,
                                          args.rtol, args.atol)
        print(f"  [{status:5s}] {name:<{width}}  {detail}")
        if status == "OK":
            n_ok += 1
        elif status == "ERROR":
            n_err += 1
            failures.append(name)
        else:
            n_fail += 1
            failures.append(name)

    print()
    print(f"Summary: {n_ok} OK, {n_fail} FAIL, {n_err} ERROR "
          f"(of {len(common)} common variables)")

    sys.exit(0 if (n_fail == 0 and n_err == 0 and not only_a and not only_b) else 1)


if __name__ == "__main__":
    main()
