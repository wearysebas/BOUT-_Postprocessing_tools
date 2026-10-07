#!/usr/bin/env python3
"""
Remove the y-guard cells from BOUT.squash.nc files made with squashoutput's
default yguards="include_upper", so they match grids without y-boundary cells
(and load in xBOUT/xHermes exactly like the raw dumps).

Layout of y in such a file (MYG guard rows, ny_inner, ny from the file):
    [MYG] [ny_inner interior] [MYG] [MYG] [ny - ny_inner interior] [MYG]
Kept rows: MYG .. MYG+ny_inner-1  and  ny_inner+3*MYG .. ny+3*MYG-1

Files whose y size is already ny are skipped. Each file is rewritten to a
temporary file, checked, then swapped in.

Usage:
    python strip_squash_yguards.py                 # default Viking roots
    python strip_squash_yguards.py DIR [DIR ...]   # any folders (searched recursively)
    python strip_squash_yguards.py --dry-run
"""

import argparse
import os
import sys

import numpy as np
from netCDF4 import Dataset

DEFAULT_ROOTS = [
    "/users/sr1937/scratch/Hermes-Final-Pro-Max/SF-Sims/No_Drifts",
    "/users/sr1937/scratch/Hermes-Final-Pro-Max/SF-Sims/CVODE_test",
]
NAME = "BOUT.squash.nc"


def scalar(ds, name):
    return int(np.asarray(ds.variables[name][...]).flatten()[0])


def keep_index(ds):
    ny, ny_inner, myg = scalar(ds, "ny"), scalar(ds, "ny_inner"), scalar(ds, "MYG")
    nyf = len(ds.dimensions["y"])
    if nyf == ny:
        return None, f"already has ny = {ny}"
    if nyf == ny + 2 * myg:
        keep = np.arange(myg, myg + ny)
    elif nyf == ny + 4 * myg:
        keep = np.r_[myg:myg + ny_inner, ny_inner + 3 * myg:ny + 3 * myg]
    else:
        raise ValueError(f"unexpected y size {nyf} (ny = {ny}, MYG = {myg})")
    assert len(keep) == ny
    return keep, f"y {nyf} -> {ny}"


def strip(path, dry_run):
    with Dataset(path) as src:
        keep, msg = keep_index(src)
        if keep is None or dry_run:
            return ("skip" if keep is None else "dry-run"), msg

        tmp = path + ".yfix.tmp"
        with Dataset(tmp, "w", format=src.data_model) as dst:
            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
            for name, dim in src.dimensions.items():
                size = None if dim.isunlimited() else (len(keep) if name == "y" else len(dim))
                dst.createDimension(name, size)
            for name, var in src.variables.items():
                attrs = {a: var.getncattr(a) for a in var.ncattrs() if a != "_FillValue"}
                if var.dtype == str:
                    out = dst.createVariable(name, str, var.dimensions)
                else:
                    filt = var.filters() or {}
                    out = dst.createVariable(name, var.dtype, var.dimensions,
                                             zlib=bool(filt.get("zlib")),
                                             complevel=filt.get("complevel") or 4)
                out.setncatts(attrs)
                if "y" in var.dimensions:
                    ax = var.dimensions.index("y")
                    data = np.take(var[...], keep, axis=ax)
                else:
                    data = var[...]
                out[...] = data

    with Dataset(path) as a, Dataset(tmp) as b:
        ok = len(b.dimensions["y"]) == len(keep) and set(a.variables) == set(b.variables)
        for name in ("Te", "Ne", "t_array"):
            if ok and name in a.variables and "y" in a.variables[name].dimensions:
                ax = a.variables[name].dimensions.index("y")
                ok = np.array_equal(np.take(a.variables[name][...], keep, axis=ax), b.variables[name][...])
    if not ok:
        os.remove(tmp)
        return "failed", "verification failed, original left untouched"
    os.replace(tmp, path)
    return "ok", msg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="*", default=DEFAULT_ROOTS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    files = sorted(os.path.join(d, NAME) for r in args.roots for d, _, fs in os.walk(r) if NAME in fs)
    if not files:
        print("No squash files found.")
        return 1

    status = {}
    for f in files:
        try:
            s, msg = strip(f, args.dry_run)
        except Exception as e:
            s, msg = "failed", repr(e)
            if os.path.exists(f + ".yfix.tmp"):
                os.remove(f + ".yfix.tmp")
        status[f] = s
        print(f"{s:8s} {f}  ({msg})")
    return 1 if "failed" in status.values() else 0


if __name__ == "__main__":
    sys.exit(main())
