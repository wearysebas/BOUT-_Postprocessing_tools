#!/usr/bin/env python3
"""
target_finder.py — INGRID target-plate factory for MAST-U snowflakes

Given a G-EQDSK (e.g. one written by freegs_sf_creator.py), locate the four
lower-divertor legs of the two X-points and write INGRID target-plate files
positioned ACROSS each leg and sized to span the full psiN range INGRID traces
(private flux psi_pf .. SOL psi_1) with margin -- so the leg traces always
intercept their plate instead of escaping to the domain boundary.

Why this exists
---------------
The hand-drawn sf15 plates fail on sf165 for two reasons:
  (1) after the primary/secondary roles swap left<->right, each plate sits on
      the wrong side of the machine, and
  (2) their psiN coverage falls ~1e-3 short of psi_pf_2 -- INGRID then reports
      'line missed the intended target ... intersected the boundary instead'.
Deriving the plates from the actual equilibrium removes both failure modes.

Plate <-> leg mapping (INGRID SF165 convention, see topologies/sf165.py):
    primary (active) null   -> WestPlate1 (its inboard leg)  + EastPlate1
    secondary null          -> WestPlate2 (its inboard leg)  + EastPlate2
"West" = the lower-R strike of the pair.  "1" = primary, "2" = secondary.

Usage
-----
    python3 target_finder.py path/to/file.geqdsk [options]

Writes W1_target.txt E1_target.txt W2_target.txt E2_target.txt + an overlay
PNG to --outdir, and prints a ready-to-paste target_plates: yaml block.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.interpolate import RectBivariateSpline

from freegs_sf_creator import parse_geqdsk, lower_xpoints, DEFAULT_WALL


# --------------------------------------------------------------------------
# psi accessors (high-order spline so the saddle geometry is resolved)
# --------------------------------------------------------------------------
class PsiMap:
    def __init__(self, d):
        self.d = d
        self.sp = RectBivariateSpline(d["R"], d["Z"], d["psi"], kx=4, ky=4)
        self.simag, self.sibry = d["simag"], d["sibry"]
        self.sgn = np.sign(self.sibry - self.simag)      # +1 if psi rises outward
        self.Rlim = (d["R"][0], d["R"][-1])
        self.Zlim = (d["Z"][0], d["Z"][-1])

    def psi(self, R, Z):
        return float(self.sp(R, Z, grid=False))

    def grad(self, R, Z):
        return np.array([float(self.sp(R, Z, dx=1, grid=False)),
                         float(self.sp(R, Z, dy=1, grid=False))])

    def hess(self, R, Z):
        return (float(self.sp(R, Z, dx=2, grid=False)),
                float(self.sp(R, Z, dx=1, dy=1, grid=False)),
                float(self.sp(R, Z, dy=2, grid=False)))

    def psin(self, R, Z):
        return (self.psi(R, Z) - self.simag) / (self.sibry - self.simag)

    def inside(self, p):
        return (self.Rlim[0] < p[0] < self.Rlim[1] and
                self.Zlim[0] < p[1] < self.Zlim[1])


# --------------------------------------------------------------------------
# separatrix-leg geometry
# --------------------------------------------------------------------------
def leg_directions(pm, xpt):
    """Four unit rays along the separatrix branches leaving the X-point
    (the asymptotes of the saddle: psi - psi_x = 0)."""
    prr, prz, pzz = pm.hess(*xpt)
    w, V = np.linalg.eigh(np.array([[prr, prz], [prz, pzz]]))
    l2, l1 = w[0], w[1]                       # l2 < 0 < l1 for a saddle
    if not (l1 > 0 > l2):
        return []
    e2, e1 = V[:, 0], V[:, 1]
    a = np.sqrt(-l2 / l1)
    v1, v2 = a * e1 + e2, -a * e1 + e2
    return [v / np.linalg.norm(v) for v in (v1, -v1, v2, -v2)]


def trace_leg(pm, xpt, direction, limiter, h=0.004, max_len=1.0, eps0=0.015):
    """Predictor-corrector march along the separatrix (psi = psi_x) from the
    X-point outward along `direction`, stopping at the limiter or domain edge.
    Returns (path[Nx2], hit_limiter_bool)."""
    psi_x = pm.psi(*xpt)
    p = np.array(xpt, float) + eps0 * np.asarray(direction)
    prev = np.asarray(direction, float)
    path, length, hit = [tuple(p)], 0.0, False
    while length < max_len and pm.inside(p):
        g = pm.grad(*p)
        g2 = g @ g
        if g2 < 1e-14:
            break
        t = np.array([g[1], -g[0]]) / np.sqrt(g2)        # tangent to contour
        if t @ prev < 0:
            t = -t
        p = p + h * t                                     # predictor
        for _ in range(3):                                # corrector -> psi_x
            g = pm.grad(*p); g2 = g @ g
            if g2 < 1e-14:
                break
            p = p - ((pm.psi(*p) - psi_x) / g2) * g
        prev = t
        length += h
        if limiter is not None and not limiter.contains_point(p):
            hit = True
            break
        path.append(tuple(p))
    return np.array(path), hit


def divertor_legs(pm, xpt, other, axis, limiter):
    """Return the (up to) two legs of `xpt` that run into the divertor:
    drop branches heading back to the core or into the other X-point.

    `other` is the partner X-point for a two-null (snowflake) config; pass
    ``None`` for a single null (LSN/USN) to skip the inter-null rejection."""
    legs = []
    a_hat = np.array(axis) - np.array(xpt)
    a_hat = a_hat / (np.linalg.norm(a_hat) + 1e-30)
    for d in leg_directions(pm, xpt):
        if d @ a_hat > 0.5:                               # points at the core
            continue
        path, _ = trace_leg(pm, xpt, d, limiter)
        end = path[-1]
        if other is not None and np.hypot(*(end - np.array(other))) < 0.06:
            continue                                      # ran into other null
        if np.hypot(*(end - np.array(axis))) < 0.30:      # curled back to core
            continue
        legs.append(path)
    # keep the two reaching furthest from the axis (region-agnostic: works for
    # both lower and upper divertors, unlike a fixed lowest-Z criterion)
    legs.sort(key=lambda pth: -np.hypot(*(np.array(pth[-1]) - np.array(axis))))
    return legs[:2]


# --------------------------------------------------------------------------
# plate construction: a segment crossing the leg, spanning [psi_pf, psi_1]
# --------------------------------------------------------------------------
def build_plate(pm, leg, psi_pf, psi_sol, margin, backoff=0.03,
                step=0.0025, max_reach=0.35, npts=9):
    """Anchor a plate `backoff` m back from the leg's wall end, orient it
    PERPENDICULAR to the local leg tangent (robust even in the weak-gradient
    zone just below the X-point, where grad(psi) is unreliable), and extend it
    until psiN spans [psi_pf - margin, psi_sol + margin].  The SOL side is
    chosen from the measured psiN, not the gradient sign."""
    # anchor a little inside the strike end so the plate stays in the domain
    s, idx = 0.0, len(leg) - 1
    while idx > 0 and s < backoff:
        s += np.hypot(*(leg[idx] - leg[idx - 1])); idx -= 1
    anchor = leg[idx].astype(float)

    # plate normal = perpendicular to the leg (finite-difference tangent)
    j, k = min(idx + 3, len(leg) - 1), max(idx - 3, 0)
    tang = leg[j] - leg[k]
    tang = tang / (np.linalg.norm(tang) + 1e-30)
    perp = np.array([tang[1], -tang[0]])
    if pm.psin(*(anchor + 0.01 * perp)) < pm.psin(*(anchor - 0.01 * perp)):
        perp = -perp                                      # +perp now climbs to SOL
    lo, hi = psi_pf - margin, psi_sol + margin

    def march(sense, stop):
        """Walk sense*perp keeping psiN monotonic; stop at `stop` or the local
        psiN ridge/valley (so inter-null legs give a short valid plate instead
        of smearing along the floor)."""
        P, prev, t = anchor.copy(), pm.psin(*anchor), 0.0
        while t < max_reach and pm.inside(P):
            nxt = P + sense * step * perp
            pv = pm.psin(*nxt)
            if sense > 0 and pv < prev - 1e-6:    # stopped climbing -> ridge
                break
            if sense < 0 and pv > prev + 1e-6:    # stopped descending -> valley
                break
            P, prev, t = nxt, pv, t + step
            if (sense > 0 and pv >= hi) or (sense < 0 and pv <= lo):
                break
        return P

    P_hi, P_lo = march(+1, hi), march(-1, lo)
    pts = np.linspace(P_lo, P_hi, npts)
    span = (pm.psin(*P_lo), pm.psin(*P_hi))
    return pts, anchor, span


def build_plate_on_limiter(pm, leg, wall, psi_pf, psi_sol, margin,
                           npts=2, ds=0.002, max_walk=0.6, reach_tol=0.02):
    """Build a plate that LIES ON the limiter (single-null route).

    `build_plate` floats a segment perpendicular to the leg ~`backoff` m inside
    the wall, so its ends straddle the limiter (one pokes through, one falls
    short) and INGRID's grid terminates off the physical boundary.  Here we
    instead carve the plate out of the wall itself: find where the leg strikes
    the limiter, then walk ALONG the limiter polyline both ways from the strike,
    SOL one way (rising psiN) and PF the other (falling psiN).

    MAXIMAL WALL ARC: each walk stops at `psi_sol + margin` (SOL) / `psi_pf -
    margin` (PF) if the wall reaches it, OR at the wall's local psiN extremum
    (ridge climbing / valley descending) if it doesn't -- so BOTH endpoints
    always sit on the limiter.  Where the wall grazes the separatrix (e.g. the
    E1 inner-divertor tile, whose PF side only dips to psiN~0.985) the plate
    simply stops at that valley; the reported span then falls short of psi_pf
    and `compute_plates` prints LOW, signalling that psi_pf_1 must be raised (or
    strike_pt_loc=limiter used) for that leg.

    Returns the same ``(pts, anchor, span)`` tuple as `build_plate`.  `npts` is
    accepted for signature compatibility but ignored: the plate follows the
    wall vertices (endpoints + any corners in span), not a fixed point count.
    Falls back to `build_plate` (with a warning) if the leg never reached the
    wall or the SOL/PF directions can't be told apart.
    """
    strike = np.asarray(leg[-1], float)
    lo, hi = psi_pf - margin, psi_sol + margin

    # densify the closed limiter into an ordered point list, flagging which
    # samples are original vertices (so the plate can follow the wall's corners)
    W = np.asarray(wall, float)
    P, VTX = [], []
    for k in range(len(W) - 1):
        a, b = W[k], W[k + 1]
        seg = b - a
        L = np.hypot(*seg)
        n = max(1, int(round(L / ds)))
        for j in range(n):                      # include start vertex at j == 0
            P.append(a + (j / n) * seg)
            VTX.append(j == 0)
    P.append(W[-1]); VTX.append(True)
    P = np.array(P)
    M = len(P)
    PSN = np.array([pm.psin(*p) for p in P])

    # locate the strike on the wall; bail to build_plate if the leg never
    # actually reached the limiter (then "on the wall" is meaningless)
    i0 = int(np.argmin(np.hypot(P[:, 0] - strike[0], P[:, 1] - strike[1])))
    if np.hypot(*(P[i0] - strike)) > reach_tol:
        print(f"  !! leg strike {tuple(np.round(strike, 4))} is "
              f">{reach_tol*1e3:.0f} mm from the limiter -- falling back to a "
              "perpendicular plate")
        return build_plate(pm, leg, psi_pf, psi_sol, margin, npts=npts)

    max_steps = int(max_walk / ds)
    eps, min_steps = 1e-4, 5            # extremum-reversal noise floor / min run

    # SOL is whichever way psiN climbs out of the strike; PF is the other way
    sol_step = +1 if PSN[(i0 + 1) % M] > PSN[(i0 - 1) % M] else -1
    if abs(PSN[(i0 + 1) % M] - PSN[(i0 - 1) % M]) < eps:
        print("  !! wall is psiN-flat at the strike -- can't separate SOL/PF, "
              "falling back to a perpendicular plate")
        return build_plate(pm, leg, psi_pf, psi_sol, margin, npts=npts)

    def march(step, target, climbing):
        """Walk the wall from i0 in `step`; stop at `target` if reached, else at
        the local psiN ridge (climbing) / valley (not climbing).  Returns
        (stop_index_on_the_i0_side, endpoint_on_wall)."""
        i = i0
        for n in range(max_steps):
            j = (i + step) % M
            if (climbing and PSN[j] >= target) or (not climbing and PSN[j] <= target):
                t = (target - PSN[i]) / (PSN[j] - PSN[i])
                return i, P[i] + t * (P[j] - P[i])          # hit the target level
            reversed_ = (PSN[j] < PSN[i] - eps if climbing
                         else PSN[j] > PSN[i] + eps)
            if reversed_ and n >= min_steps:
                return i, P[i]                              # wall extremum
            i = j
        return i, P[i]                                      # ran out of wall arc

    i_sol, p_sol = march(sol_step, hi, climbing=True)
    i_pf, p_pf = march(-sol_step, lo, climbing=False)

    # walk the densified indices PF -> strike -> SOL, keeping the original wall
    # vertices among them so the plate follows the wall's corners
    if sol_step > 0:                                # PF at lower index, SOL higher
        seq = range(i_pf + 1, i_sol + 1)
    else:                                           # PF at higher index, SOL lower
        seq = range(i_pf - 1, i_sol - 1, -1)
    interior = [P[i] for i in seq if VTX[i]]
    pts = np.array([p_pf, *interior, p_sol])
    span = (pm.psin(*p_pf), pm.psin(*p_sol))
    return pts, strike, span


def write_plate(fn, pts):
    with open(fn, "w") as f:
        for R, Z in pts:
            f.write(f"{R:.6f}, {Z:.6f}\n")


# --------------------------------------------------------------------------
# reusable plate factory + overlay (shared with eqdsk_to_ingrid.py)
# --------------------------------------------------------------------------
def compute_plates(pm, axis, jobs, psi_sol, margin, limiter, outdir, npts=2,
                   wall=None, on_limiter=False):
    """Trace divertor legs and write INGRID target-plate files.

    `jobs` is an iterable of ``(null, other_or_None, tag, psi_pf)`` tuples; for
    each X-point its two divertor legs become ``W<tag>`` / ``E<tag>`` plates
    spanning psiN ``[psi_pf - margin, psi_sol + margin]``.  Pass ``other=None``
    for a single-null config.  Writes ``<name>_target.txt`` into `outdir` and
    returns ``{name: (pts, anchor, span, leg)}``.

    `npts` is the number of points written per plate.  Per INGRID's developer,
    a plate only needs TWO points to make a straight line that crosses the flux
    surfaces, so the default is 2 (just the [psi_pf, psi_sol] endpoints); raise
    it for a curved plate that follows the leg's local geometry more closely.

    With ``on_limiter=True`` (and `wall` given, the ordered limiter polygon),
    plates are instead carved out of the limiter so the grid ends on the
    physical boundary -- see `build_plate_on_limiter`.  Used for the single-null
    route; the default (False) leaves the snowflake plates unchanged.
    """
    plates = {}
    for null, other, tag, psi_pf in jobs:
        legs = divertor_legs(pm, null, other, axis, limiter)
        if len(legs) < 2:
            print(f"  !! only {len(legs)} divertor leg(s) found for null {tag} "
                  f"({null[0]:.3f},{null[1]:.3f}) -- check the overlay")
        legs.sort(key=lambda pth: pth[-1][0])             # West = lower-R strike
        for leg, we in zip(legs, ("W", "E")):
            if on_limiter and wall is not None:
                pts, anchor, span = build_plate_on_limiter(
                    pm, leg, wall, psi_pf, psi_sol, margin, npts=npts)
            else:
                pts, anchor, span = build_plate(pm, leg, psi_pf, psi_sol,
                                                margin, npts=npts)
            name = f"{we}{tag}"
            plates[name] = (pts, anchor, span, leg)
            fn = outdir / f"{name}_target.txt"
            write_plate(fn, pts)
            ok = "ok " if span[0] <= psi_pf and span[1] >= psi_sol else "LOW"
            print(f"  {name}: strike R={anchor[0]:.4f} Z={anchor[1]:+.4f}  "
                  f"psiN span [{span[0]:.4f}, {span[1]:.4f}]  {ok}  -> {fn.name}")
    return plates


def plot_overlay(d, wall, nulls_labeled, plates, psi_sol, outdir, title):
    """Save a psiN/plate overlay PNG.  `nulls_labeled` is a list of
    ``((R, Z), label)`` for the X-points; `plates` is the compute_plates dict."""
    pn = (d["psi"] - d["simag"]) / (d["sibry"] - d["simag"])
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.contour(d["R"], d["Z"], pn.T, levels=np.linspace(0.9, 0.99, 4),
               colors="0.75", linewidths=0.5)
    ax.contour(d["R"], d["Z"], pn.T, levels=[1.0], colors="r", linewidths=1.3)
    ax.contour(d["R"], d["Z"], pn.T, levels=[psi_sol], colors="C1",
               linewidths=0.6)
    ax.plot(wall[:, 0], wall[:, 1], "k--", lw=1)
    for x, lab in nulls_labeled:
        ax.plot(*x, "kx", ms=10, mew=2)
        ax.annotate(lab, x, textcoords="offset points", xytext=(5, -11), fontsize=8)
    cols = {"W1": "b", "E1": "g", "W2": "m", "E2": "c"}
    for name, (pts, anchor, span, leg) in plates.items():
        ax.plot(leg[:, 0], leg[:, 1], "-", color=cols.get(name, "k"), lw=0.8, alpha=0.5)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=cols.get(name, "k"), lw=2.5, label=name)
    # auto-zoom around the strike region (works for lower and upper divertors)
    if plates:
        allp = np.vstack([np.vstack([pts, leg])
                          for (pts, _, _, leg) in plates.values()])
        rc, zc = allp[:, 0], allp[:, 1]
        ax.set_xlim(rc.min() - 0.15, rc.max() + 0.15)
        ax.set_ylim(zc.min() - 0.15, zc.max() + 0.15)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(title)
    png = outdir / "target_finder.png"
    plt.tight_layout(); plt.savefig(png, dpi=140); plt.close(fig)
    print(f"  saved overlay -> {png}")
    return png


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Derive INGRID target plates (W1/E1/W2/E2) from a snowflake "
                    "G-EQDSK by tracing the four divertor legs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("eqdsk", help="input G-EQDSK file")
    ap.add_argument("--limiter", default=str(DEFAULT_WALL),
                    help="limiter polygon (R,Z); legs stop here and plates "
                         "anchor just inside it -- pass the SAME file INGRID grids with")
    ap.add_argument("--psi-pf1", type=float, default=0.98,
                    help="primary private-flux psiN (yaml psi_pf_1)")
    ap.add_argument("--psi-pf2", type=float, default=0.985,
                    help="secondary private-flux psiN (yaml psi_pf_2)")
    ap.add_argument("--psi-sol", type=float, default=1.032,
                    help="SOL boundary psiN the legs trace to (yaml psi_1)")
    ap.add_argument("--margin", type=float, default=0.012,
                    help="extra psiN each side so traces hit with room to spare")
    ap.add_argument("--primary", choices=("auto", "left", "right"), default="auto",
                    help="which lower null is the active/primary one (snowflake)")
    ap.add_argument("--single-null", action="store_true",
                    help="LSN: build W1/E1 only from the single lower X-point "
                         "(do not require a second null)")
    ap.add_argument("--plate-points", type=int, default=2,
                    help="points per plate; 2 = the developer's straight-line "
                         "route (just the psi_pf/psi_sol endpoints)")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default <eqdsk dir>/target_plates_auto)")
    args = ap.parse_args()

    d = parse_geqdsk(args.eqdsk)
    pm = PsiMap(d)
    axis = (d["rmaxis"], d["zmaxis"])

    wall = np.loadtxt(args.limiter, delimiter=",")
    limiter = MplPath(wall)

    xs = lower_xpoints(d, ntop=1 if args.single_null else 2)
    print(f"eqdsk      : {args.eqdsk}")
    print(f"limiter    : {args.limiter}")
    print(f"mag axis   : R={axis[0]:.4f} Z={axis[1]:+.4f}")

    outdir = Path(args.outdir) if args.outdir else Path(args.eqdsk).resolve().parent / "target_plates_auto"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.single_null:
        if len(xs) < 1:
            raise SystemExit(f"need one lower X-point, found {len(xs)}: {xs}")
        primary = xs[0]
        print(f"primary  X : R={primary[0]:.4f} Z={primary[1]:+.4f}  "
              f"psiN={pm.psin(*primary):.4f}  (single-null)")
        jobs = ((primary, None, "1", args.psi_pf1),)
        labels = [(primary, "X")]
    else:
        if len(xs) < 2:
            raise SystemExit(f"need two lower X-points, found {len(xs)}: {xs} "
                             "(pass --single-null for an LSN)")
        xs.sort(key=lambda p: p[0])                        # inboard (left) first
        left, right = xs[0], xs[1]
        if args.primary == "left":
            primary, secondary = left, right
        elif args.primary == "right":
            primary, secondary = right, left
        else:                                              # auto: psiN closest to 1
            primary, secondary = (
                (right, left) if abs(pm.psin(*right) - 1) <= abs(pm.psin(*left) - 1)
                else (left, right))
        print(f"primary  X : R={primary[0]:.4f} Z={primary[1]:+.4f}  psiN={pm.psin(*primary):.4f}")
        print(f"secondary X: R={secondary[0]:.4f} Z={secondary[1]:+.4f}  psiN={pm.psin(*secondary):.4f}")
        jobs = ((primary, secondary, "1", args.psi_pf1),
                (secondary, primary, "2", args.psi_pf2))
        labels = [(primary, "PRIM"), (secondary, "SEC")]

    plates = compute_plates(pm, axis, jobs, args.psi_sol, args.margin,
                            limiter, outdir, npts=args.plate_points,
                            wall=wall, on_limiter=args.single_null)

    plot_overlay(d, wall, labels, plates,
                 args.psi_sol, outdir, f"target_finder: {Path(args.eqdsk).name}")

    # ---- ready-to-paste yaml ------------------------------------------
    print("\n  target_plates yaml block (paths relative to the yaml dir):")
    print("  target_plates:")
    for name in ("E1", "E2", "W1", "W2"):
        if name in plates:
            print(f"    plate_{name}:\n      file: target_plates_auto/{name}_target.txt")


if __name__ == "__main__":
    main()
