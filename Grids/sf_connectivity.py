import sys
import numpy as np
import xarray as xr


def load(path):
    ds = xr.open_dataset(path)
    g = {k: int(ds[k].values) for k in
         ("nx", "ny", "ixseps1", "ixseps2", "jyseps1_1", "jyseps2_1",
          "jyseps1_2", "jyseps2_2", "ny_inner")}
    g["rm"], g["zm"] = ds["rm"].values, ds["zm"].values
    return g


def face(g, i, j, which):
    ip = j + 1 if j < g["ny_inner"] else j + 3
    jr = i - 1
    idx = [2, 4] if which == "up" else [1, 3]
    return np.column_stack([g["rm"][ip, jr, idx], g["zm"][ip, jr, idx]])


def ranges(xs):
    out, start = [], None
    for k, x in enumerate(xs):
        if start is None:
            start = x
        if k == len(xs) - 1 or xs[k + 1] != x + 1:
            out.append(f"{start}-{x}" if x != start else f"{start}")
            start = None
    return ",".join(out)


def connectivity(path, name):
    g = load(path)
    nx, ny, nyi = g["nx"], g["ny"], g["ny_inner"]
    cuts = sorted({g["jyseps1_1"], g["jyseps2_1"], g["jyseps1_2"], nyi - 1, g["jyseps2_2"], ny - 1})
    lows = sorted({0} | {c + 1 for c in cuts if c + 1 < ny})
    xs = range(2, nx - 2)
    tol = 1e-6
    print(f"\n=== {name}: ixseps1={g['ixseps1']} ixseps2={g['ixseps2']} "
          f"j11={g['jyseps1_1']} j21={g['jyseps2_1']} j12={g['jyseps1_2']} "
          f"ny_inner={nyi} j22={g['jyseps2_2']} ny={ny}  (real x = 2..{nx - 3})")
    for yu in cuts:
        dest = {}
        for i in xs:
            up = face(g, i, yu, "up")
            hit = "target/none"
            for yl in lows:
                lo = face(g, i, yl, "down")
                if np.max(np.hypot(*(up - lo).T)) < tol:
                    hit = f"y={yl}"
                    break
            else:
                for yl in lows:
                    for i2 in xs:
                        lo = face(g, i2, yl, "down")
                        if np.max(np.hypot(*(up - lo).T)) < tol:
                            hit = f"y={yl} (x->{i2}!)"
                            break
                    if hit != "target/none":
                        break
            dest.setdefault(hit, []).append(i)
        desc = "; ".join(f"x {ranges(v)} -> {k}" for k, v in dest.items())
        print(f"  top of y={yu:3d}: {desc}")


for arg in sys.argv[1:]:
    path, name = arg.split("=", 1)[1], arg.split("=", 1)[0]
    connectivity(path, name)
