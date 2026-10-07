#!/usr/bin/env python3
"""
Squash BOUT++/Hermes-3 runs into one thinned, compressed file per run.

Walks the given root folders, finds every directory with BOUT.dmp.0.nc,
and writes <run>/BOUT.squash_with_guard_cells.nc (y-guard cells kept, upper
targets included) and <run>/BOUT.squash.nc (same data, y-guard cells stripped)
keeping every --step-th output. The time
offset is chosen so the LAST complete time slice is always kept. Ranks
with a truncated time axis (e.g. a run killed by the quota mid-write)
are handled by truncating to the shortest rank.

Usage (on Viking):
    python squash_runs.py                       # both default roots
    python squash_runs.py --step 50 --dry-run   # see what would happen
    python squash_runs.py --delete-dumps        # remove BOUT.dmp.* after a verified squash
"""

import argparse
import glob
import os
import shutil
import sys
import time

import numpy as np
from netCDF4 import Dataset
from boutdata.squashoutput import squashoutput
from strip_squash_yguards import strip

DEFAULT_ROOTS = [
    "/users/sr1937/scratch/Hermes-Final-Pro-Max/SF-Sims/No_Drifts",
    "/users/sr1937/scratch/Hermes-Final-Pro-Max/SF-Sims/CVODE_test",
]
OUTNAME = "BOUT.squash.nc"
GUARDNAME = "BOUT.squash_with_guard_cells.nc"


def find_runs(roots):
    runs = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"[warn] root not found: {root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            if "BOUT.dmp.0.nc" in filenames:
                runs.append(dirpath)
    return sorted(runs)


def dump_files(run):
    return sorted(glob.glob(os.path.join(run, "BOUT.dmp.*.nc")))


def complete_length(ds):
    """Number of time slices fully written: a run killed mid-write (e.g. by the
    quota) can leave a last slice where t_array or other variables are unset."""
    t = ds.variables["t_array"][:]
    tvars = [v for v in ds.variables.values() if v.dimensions[:1] == ("t",)]
    n = len(t)
    while n > 0 and (np.ma.is_masked(t[n - 1]) or any(np.ma.is_masked(v[n - 1]) for v in tvars)):
        n -= 1
    return n


def time_lengths(files):
    """Number of complete time slices in each rank's dump file (None if unreadable)."""
    lengths = {}
    for f in files:
        try:
            with Dataset(f) as ds:
                lengths[f] = complete_length(ds)
        except Exception as e:
            print(f"    [error] cannot read {os.path.basename(f)}: {e}")
            lengths[f] = None
    return lengths


def size_gb(paths):
    return sum(os.path.getsize(p) for p in paths) / 1024**3


def squash_complete(path):
    try:
        with Dataset(path) as ds:
            t = ds.variables["t_array"][:] if "t_array" in ds.variables else None
            return ("squashoutput_version" in ds.ncattrs() and t is not None
                    and len(t) > 0 and not np.ma.is_masked(t))
    except Exception:
        return False


def to_single_precision(path, complevel):
    """Rewrite float64 variables as float32. Done here rather than via
    squashoutput(singleprecision=True), which tries to cast string
    variables such as mesh_topology = "sf" and crashes."""
    tmp = path + ".f32.tmp"
    with Dataset(path) as src, Dataset(tmp, "w", format="NETCDF4") as dst:
        dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
        for name, var in src.variables.items():
            attrs = {a: var.getncattr(a) for a in var.ncattrs() if a != "_FillValue"}
            if var.dtype == "float64":
                out = dst.createVariable(name, "f4", var.dimensions, zlib=True, complevel=complevel)
            elif var.dtype == str:
                out = dst.createVariable(name, str, var.dimensions)
            else:
                out = dst.createVariable(name, var.dtype, var.dimensions, zlib=True, complevel=complevel)
            out.setncatts(attrs)
            out[...] = var[...]
    os.replace(tmp, path)


def make_stripped(run):
    src, dst = os.path.join(run, GUARDNAME), os.path.join(run, OUTNAME)
    tmp = dst + ".copy.tmp"
    try:
        shutil.copyfile(src, tmp)
        status, msg = strip(tmp, False)
    except Exception as e:
        status, msg = "failed", repr(e)
    if status not in ("ok", "skip"):
        print(f"    [fail] {GUARDNAME} -> {OUTNAME}: {msg}")
        for f in (tmp, tmp + ".yfix.tmp"):
            if os.path.exists(f):
                os.remove(f)
        return False
    os.replace(tmp, dst)
    print(f"    [ok] {OUTNAME} written from {GUARDNAME} ({msg})")
    return True


def finish(run, files, delete_dumps):
    if not make_stripped(run):
        return "failed"
    if delete_dumps:
        freed = size_gb(files)
        for f in files:
            os.remove(f)
        print(f"    deleted {len(files)} dump files, freed {freed:.2f} GB")
    return "ok"


def squash_run(run, step, single, complevel, parallel, overwrite, dry_run, delete_dumps):
    files = dump_files(run)
    outpath = os.path.join(run, GUARDNAME)
    print(f"\n=== {run}")
    print(f"    {len(files)} dump files, {size_gb(files):.2f} GB")

    if os.path.exists(outpath) and not overwrite:
        if squash_complete(outpath):
            if squash_complete(os.path.join(run, OUTNAME)):
                print(f"    [skip] {GUARDNAME} and {OUTNAME} already exist and are complete (use --overwrite)")
                return "skipped"
            print(f"    [strip] {GUARDNAME} is complete, only {OUTNAME} needs writing")
            if dry_run:
                return "dry-run"
            return finish(run, files, delete_dumps)
        print(f"    [redo] {GUARDNAME} is incomplete (job killed while writing?)")

    lengths = time_lengths(files)
    if any(n is None for n in lengths.values()):
        print("    [fail] at least one dump file is unreadable, not squashing")
        return "failed"

    nts = set(lengths.values())
    nt = min(nts)
    if len(nts) > 1:
        print(f"    [warn] ranks disagree on time length {sorted(nts)}; truncating to {nt}")
    if nt == 0:
        print("    [fail] no time slices")
        return "failed"

    start = (nt - 1) % step
    tind = slice(start, nt, step)
    n_kept = len(range(start, nt, step))
    print(f"    nt = {nt}, keeping {n_kept} slices (indices {start}..{nt - 1} step {step})")

    if dry_run:
        return "dry-run"

    if os.path.exists(outpath):
        os.remove(outpath)

    t0 = time.time()
    try:
        squashoutput(
            run,
            outputname=GUARDNAME,
            tind=tind,
            yguards="include_upper",
            compress=True,
            complevel=complevel,
            parallel=parallel,
            quiet=True,
        )
        if single:
            to_single_precision(outpath, complevel)
    except Exception as e:
        print(f"    [fail] squash raised: {e!r}")
        for f in (outpath, outpath + ".f32.tmp"):
            if os.path.exists(f):
                os.remove(f)
        return "failed"

    # Verify: right number of slices and the final time matches the full run
    with Dataset(files[0]) as ds:
        t_last_full = float(ds.variables["t_array"][nt - 1])
    with Dataset(outpath) as ds:
        t_sq = ds.variables["t_array"][:]
        ok = len(t_sq) == n_kept and abs(float(t_sq[-1]) - t_last_full) <= 1e-6 * max(1.0, abs(t_last_full))

    print(
        f"    wrote {GUARDNAME}: {size_gb([outpath]):.3f} GB, "
        f"t = {float(t_sq[0]):.4g} .. {float(t_sq[-1]):.4g} ({time.time() - t0:.0f} s)"
    )
    if not ok:
        print(f"    [fail] verification failed (expected {n_kept} slices ending at t = {t_last_full:.6g})")
        os.remove(outpath)
        return "failed"
    print("    [ok] verified")

    return finish(run, files, delete_dumps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="*", default=DEFAULT_ROOTS, help="folders to search for runs")
    p.add_argument("--step", type=int, default=100, help="keep every N-th output (default 100)")
    p.add_argument("--double", action="store_true", help="keep double precision (default: single)")
    p.add_argument("--complevel", type=int, default=4, help="zlib compression level (default 4)")
    p.add_argument("--parallel", type=int, default=0, help="worker processes for squashoutput (0 = serial)")
    p.add_argument("--overwrite", action="store_true", help="redo runs that already have a squash file")
    p.add_argument("--dry-run", action="store_true", help="only report what would be done")
    p.add_argument("--delete-dumps", action="store_true",
                   help="delete BOUT.dmp.*.nc after a verified squash (restart files are kept)")
    args = p.parse_args()

    runs = find_runs(args.roots)
    if not runs:
        print("No runs found.")
        return 1
    print(f"Found {len(runs)} run(s)")

    results = {}
    for run in runs:
        results[run] = squash_run(
            run,
            step=args.step,
            single=not args.double,
            complevel=args.complevel,
            parallel=args.parallel or False,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            delete_dumps=args.delete_dumps,
        )

    print("\n=== Summary")
    for run, status in results.items():
        print(f"    {status:8s} {run}")
    return 1 if "failed" in results.values() else 0


if __name__ == "__main__":
    sys.exit(main())
