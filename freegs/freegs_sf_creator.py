#!/usr/bin/env python3
"""
freegs_sf_creator.py — MAST-U snowflake G-EQDSK factory (FreeGS)

Generate a lower-divertor snowflake equilibrium for MAST-U with the two
lower X-points placed at user-requested positions, keeping the experimental
p'(psi) and ff'(psi) profiles (and core LCFS shape) of a reference shot.
Intended to produce families of smoothly-varying eqdsk files for INGRID
grid generation (snowflake-transition studies).

Usage
-----
    python3 freegs_sf_creator.py RXPT1 ZXPT1 [RXPT2 ZXPT2] [options]

    RXPT1 ZXPT1 : primary (active) X-point  [m]
    RXPT2 ZXPT2 : secondary X-point [m]; omit both for a single-null (LSN)

Single-null (LSN) example -- one lower X-point, no secondary:

    python3 freegs_sf_creator.py 0.6733 -1.2624 --out mastu_lsn_freegs.geqdsk

Fidelity test (reproduce the INGRID-validated reference file
g049467.070005_modified from its own X-points):

    python3 freegs_sf_creator.py 0.6733 -1.2624 0.6812 -1.3219

The reference equilibrium supplies:
  * p'(psiN) and ff'(psiN) profiles (tabulated, psiN in [0,1])
  * fvac = R*Bt (edge fpol)
  * the core LCFS shape (sampled as isoflux constraints)
  * divertor leg / strike-point anchors (see REF_ANCHORS_*)
  * the upper-half psi map, grafted onto the solution (see
    graft_reference_upper) — the reference's upper half was hand-modified
    by its author to remove the upper X-point, and INGRID requires that
The grid/domain matches the reference file (65x65, R in [0.06,2.0],
Z in [-2.2,2.2]) and the output is written in the reference's sign
conventions (psi increasing outward, fpol < 0).

Validated 2026-06-11: the fidelity-test output reproduces the reference
X-points to <1 mm and runs through the full INGRID chain (SF75 topology,
27 patches, CreateSubgrid, ExportGridue) with the mastu_sfexact.yml
settings, producing a 68x30 gridue identical in dimensions to the
reference's.  Note INGRID_Final has a bug in ExportGridue (missing CDN
import); shim with `INGRID.ingrid.CDN = INGRID.ingrid.UDN` before calling.

Near-double-null references (2026-06-11, g049465.084044 / SF75): if the
reference keeps its upper X-point close to the separatrix (psiN ~ 1.007
there, vs the hand-deleted upper null of the original reference), asking
stage A for an X-point even a few mm off the file's natural null throws
Picard into a limit cycle — the boundary flips between the lower null and
the upper X-point (Ip oscillating 0.6<->1.0 MA, period ~5) and never
converges.  Stage A therefore always solves at the reference's own
detected primary null; stage B then walks to the requested targets
(--steps continuation increments for targets far from the nulls).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freegs
from freegs import geqdsk
import freegs.critical as critical
from freegs.machine import Coil, Circuit, Solenoid, Machine
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage import minimum_filter

HERE = Path(__file__).resolve().parent
DEFAULT_REF = HERE.parent / "freegs" / "g049467.070005_modified"
DEFAULT_WALL = HERE.parent / "freegs" / "target_plates" / "simplified_limiter_mastu.txt"


# --------------------------------------------------------------------------
# G-EQDSK parsing (raw arrays — full control over profiles)
# --------------------------------------------------------------------------
def parse_geqdsk(fn):
    lines = open(fn).read().split("\n")
    head = lines[0]
    m = re.findall(r"-?\d+", head)
    nw, nh = int(m[-2]), int(m[-1])
    tk = re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+|[-+]?\d+", " ".join(lines[1:]))
    v = [float(t) for t in tk]
    i = [0]

    def take(n):
        a = np.array(v[i[0]:i[0] + n]); i[0] += n; return a

    rdim, zdim, rcentr, rleft, zmid = take(5)
    rmaxis, zmaxis, simag, sibry, bcentr = take(5)
    cur = take(5); take(5)
    fpol = take(nw); pres = take(nw); ffprim = take(nw); pprime = take(nw)
    psirz = take(nw * nh).reshape(nh, nw).T            # -> psi[iR, iZ]
    qpsi = take(nw)
    nb = int(v[i[0]]); li = int(v[i[0] + 1]); i[0] += 2
    bdry = take(2 * nb).reshape(nb, 2)
    lim = take(2 * li).reshape(li, 2)
    R = np.linspace(rleft, rleft + rdim, nw)
    Z = np.linspace(zmid - zdim / 2.0, zmid + zdim / 2.0, nh)
    return dict(head=head, nw=nw, nh=nh, R=R, Z=Z, psi=psirz,
                rleft=rleft, rdim=rdim, zmid=zmid, zdim=zdim, rcentr=rcentr,
                rmaxis=rmaxis, zmaxis=zmaxis, simag=simag, sibry=sibry,
                bcentr=bcentr, Ip=cur[0], fpol=fpol, pres=pres,
                ffprim=ffprim, pprime=pprime, qpsi=qpsi, bdry=bdry, lim=lim)


def lower_xpoints(d, window=(0.4, 1.0, -1.6, -1.05), ntop=2):
    """Return the `ntop` lower-divertor X-points nearest to psi_bndry as (R,Z)
    sorted by R (inboard first).  High-resolution quartic-spline saddle search
    because the grid-based finder misses shallow snowflake saddles."""
    R, Z, psi = d["R"], d["Z"], d["psi"]
    sp = RectBivariateSpline(R, Z, psi, kx=4, ky=4)
    r0, r1, z0, z1 = window
    Rf = np.linspace(max(r0, R[0]), min(r1, R[-1]), 240)
    Zf = np.linspace(max(z0, Z[0]), min(z1, Z[-1]), 240)
    g = np.hypot(sp(Rf, Zf, dx=1), sp(Rf, Zf, dy=1))
    mn = minimum_filter(g, size=7)
    cand = np.argwhere((g == mn) & (g < np.percentile(g, 10)))
    found = []
    for ir, iz in cand:
        rr, zz = Rf[ir], Zf[iz]
        for _ in range(50):
            gr = sp(rr, zz, dx=1)[0, 0]; gz = sp(rr, zz, dy=1)[0, 0]
            grr = sp(rr, zz, dx=2)[0, 0]; gzz = sp(rr, zz, dy=2)[0, 0]
            grz = sp(rr, zz, dx=1, dy=1)[0, 0]
            try:
                st = np.linalg.solve([[grr, grz], [grz, gzz]], [gr, gz])
            except np.linalg.LinAlgError:
                break
            rr -= st[0]; zz -= st[1]
            if not (R[0] < rr < R[-1] and Z[0] < zz < Z[-1]):
                break
            if np.hypot(*st) < 1e-8:
                break
        if r0 < rr < r1 and z0 < zz < z1:
            grr = sp(rr, zz, dx=2)[0, 0]; gzz = sp(rr, zz, dy=2)[0, 0]
            grz = sp(rr, zz, dx=1, dy=1)[0, 0]
            if grr * gzz - grz ** 2 < 0:     # saddle = X-point
                found.append((float(rr), float(zz), float(sp(rr, zz)[0, 0])))
    uniq = []
    for p in found:
        if not any(abs(p[0] - q[0]) < 1e-2 and abs(p[1] - q[1]) < 1e-2
                   for q in uniq):
            uniq.append(p)
    uniq.sort(key=lambda p: abs(p[2] - d["sibry"]))
    uniq = uniq[:ntop]
    uniq.sort(key=lambda p: p[0])            # inboard first
    return [(p[0], p[1]) for p in uniq]


def write_geqdsk_dict(d, fn):
    """Write a G-EQDSK file from a parsed dict (same layout as parse_geqdsk)."""
    f = open(fn, "w")
    f.write(f"{d['head'][:48]:48s}{0:4d}{d['nw']:4d}{d['nh']:4d}\n")

    co = [0]
    def fmt(*vals):
        for v in vals:
            f.write(f"{v:16.9E}")
            co[0] += 1
            if co[0] % 5 == 0:
                f.write("\n")
    def end():
        if co[0] % 5 != 0:
            f.write("\n")
        co[0] = 0

    fmt(d["rdim"], d["zdim"], d["rcentr"], d["rleft"], d["zmid"])
    fmt(d["rmaxis"], d["zmaxis"], d["simag"], d["sibry"], d["bcentr"])
    fmt(d["Ip"], d["simag"], 0.0, d["rmaxis"], 0.0)
    fmt(d["zmaxis"], 0.0, d["sibry"], 0.0, 0.0)
    end()
    for arr in ("fpol", "pres", "ffprim", "pprime"):
        fmt(*d[arr]); end()
    fmt(*d["psi"].T.flatten()); end()      # psi[iR,iZ] -> file order (Z-major)
    fmt(*d["qpsi"]); end()
    f.write(f"{len(d['bdry']):5d}{len(d['lim']):5d}\n")
    fmt(*d["bdry"].flatten()); end()
    fmt(*d["lim"].flatten()); end()
    f.close()


def match_reference_signs(new, ref):
    """Flip the output's sign conventions to the reference's (COCOS match).

    FreeGS writes psi DEcreasing outward with positive fpol; the reference
    has psi INcreasing outward with negative fpol (Bt in -phi).  INGRID's
    gridue export checks Bp orientation against the grid ('Bp_dot_grady and
    Bpxy have opposite signs') and rejects the FreeGS convention, so flip
    psi, fpol, bcentr and the profile derivatives.  q is invariant under the
    combined flip."""
    flip_psi = (new["sibry"] - new["simag"]) * (ref["sibry"] - ref["simag"]) < 0
    if flip_psi:
        for k in ("psi", "simag", "sibry", "pprime", "ffprim"):
            new[k] = -new[k]
    if new["fpol"][-1] * ref["fpol"][-1] < 0:
        new["fpol"] = -new["fpol"]
        new["bcentr"] = -new["bcentr"]
    return new


def graft_reference_upper(new, ref, z0=0.1, z1=0.6):
    """Replace the upper region of the solved psi map with the reference's.

    The reference file's upper half was hand-modified by its author to remove
    the upper X-point (the psi map core tops at Z~0.5 while its bdry array
    claims Z~1.12) — and INGRID demonstrably accepts it.  Grafting that exact
    upper region onto every member of the sweep (i) removes our solve's
    natural upper null at psiN~1.002 and (ii) keeps the upper half IDENTICAL
    across the family, so only the divertor varies between grids.

    The reference psi is first mapped to the solve's gauge with the affine
    transform that matches axis and boundary flux (exact: both maps share the
    same magnetic-axis position and X-point-defined separatrix).  Blend is a
    cosine ramp in Z over [z0, z1]: pure solve below z0, pure reference
    above z1.
    """
    a = (new["simag"] - new["sibry"]) / (ref["simag"] - ref["sibry"])
    b = new["simag"] - a * ref["simag"]
    psi_ref = a * ref["psi"] + b

    Z = new["Z"]
    w = np.ones_like(Z)                       # weight of the SOLVED psi
    band = (Z > z0) & (Z < z1)
    w[band] = 0.5 * (1.0 + np.cos(np.pi * (Z[band] - z0) / (z1 - z0)))
    w[Z >= z1] = 0.0
    new["psi"] = w[np.newaxis, :] * new["psi"] + \
        (1.0 - w[np.newaxis, :]) * psi_ref
    return new


# --------------------------------------------------------------------------
# Profiles from the reference file's p' and ff'
# --------------------------------------------------------------------------
def make_profiles(d, psi_sign):
    """psi_sign = +1 or -1 matches FreeGS' internal psi orientation.
    pprime/ffprim in the file are tabulated on uniform psi_norm in [0,1]."""
    psin = np.linspace(0.0, 1.0, d["nw"])
    pp = interp1d(psin, d["pprime"] * psi_sign, bounds_error=False,
                  fill_value=(d["pprime"][0] * psi_sign, 0.0))
    ff = interp1d(psin, d["ffprim"] * psi_sign, bounds_error=False,
                  fill_value=(d["ffprim"][0] * psi_sign, 0.0))
    fvac = d["fpol"][-1]                    # R*Bt in vacuum (edge value)
    return freegs.jtor.ProfilesPprimeFfprime(pp, ff, fvac)


# --------------------------------------------------------------------------
# Machine (MAST-U).  Configuration validated against g049467.070005_modified
# (2026-06-10):
#   * Lower divertor coils (D1L..D7L, DpL) independently controlled — they
#     sculpt the snowflake.
#   * Upper divertor coils frozen at 0 A: if controllable, the regularised
#     solution drifts toward a connected double-null (upper X at psiN=1.000
#     steals the separatrix from the snowflake nulls).
#   * P4, P5 wired as up-down SYMMETRIC circuits (real machine wiring) —
#     equilibrium vertical field without sculpting an upper null.
#   * P6 vertical-control circuits frozen at 0: when controllable the
#     least-squares solution drives them to +-50..75 kA (they are 2-turn
#     coils!) fighting each other and spawning nulls near (1.2, -1.1).
#   * Px circuit frozen at +1000 A.  +2000 A produces a spurious O/X pair at
#     (R~0.30, Z~-1.24), psiN~1.01, in the inner-leg path to the W1 target
#     (breaks INGRID leg tracing); 0 A hands the boundary to the upper
#     region.  +1000 A gives a clean null census and psiN(X1,X2) =
#     1.0000/1.0003 with physical (few-kA) coil currents.
# --------------------------------------------------------------------------
def build_machine(wall_file=DEFAULT_WALL, px=1000.0, freeze_lower_d=False):
    # (name, R, Z, turns) for each upper coil; lower is the Z-mirror.
    specs = [("D1", 0.381, 1.555, 35), ("D2", 0.574, 1.734, 23),
             ("D3", 0.815, 1.980, 23), ("Dp", 0.918, 1.501, 23),
             ("D5", 1.900, 1.950, 27), ("D6", 1.285, 1.470, 23),
             ("D7", 1.520, 1.470, 23)]
    coils = [("Solenoid", Solenoid(0.19475, -1.581, 1.581, 324)),
             ("Pc", Solenoid(0.067, -0.6, 0.6, 142))]
    for nm, R, Z, t in specs:
        coils.append((nm + "U", Coil(R,  Z, turns=t)))
        coils.append((nm + "L", Coil(R, -Z, turns=t)))
    coils.append(("P4", Circuit([("P4U", Coil(1.500, 1.100, turns=23), 1.0),
                                 ("P4L", Coil(1.500, -1.100, turns=23), 1.0)])))
    coils.append(("P5", Circuit([("P5U", Coil(1.650, 0.357, turns=23), 1.0),
                                 ("P5L", Coil(1.650, -0.357, turns=23), 1.0)])))
    coils.append(("Px", Circuit([("PxU", Coil(0.2405, 1.2285, turns=44), 1.0),
                                 ("PxL", Coil(0.2405, -1.2285, turns=44), 1.0)])))
    coils.append(("P61", Circuit([("P61U", Coil(1.1975, 1.11175, turns=2), 1.0),
                                  ("P61L", Coil(1.1975, -1.11175, turns=2), -1.0)])))
    coils.append(("P62", Circuit([("P62U", Coil(1.2575, 1.0575, turns=2), 1.0),
                                  ("P62L", Coil(1.2575, -1.0575, turns=2), -1.0)])))
    tok = Machine(coils)
    tok["Px"].control = False
    tok["Px"].current = px
    for c in ("P61", "P62"):
        tok[c].control = False
        tok[c].current = 0.0
    for c in ("D1U", "D2U", "D3U", "DpU", "D5U", "D6U", "D7U"):
        tok[c].control = False
        tok[c].current = 0.0
    # For a single null (LSN) the lower divertor coils are what sculpt the
    # snowflake's SECOND lower null; freezing them at 0 removes that degree of
    # freedom so the solve yields a single lower X-point (shape then set by
    # P4/P5/Px + plasma).
    if freeze_lower_d:
        for c in ("D1L", "D2L", "D3L", "DpL", "D5L", "D6L", "D7L"):
            tok[c].control = False
            tok[c].current = 0.0
    wall = np.loadtxt(wall_file, delimiter=",")
    tok.wall = freegs.machine.Wall(wall[:, 0], wall[:, 1])
    return tok


# --------------------------------------------------------------------------
# Separatrix leg anchors, extracted from g049467.070005_modified (psiN=1.0
# crossings).  Without them the unconstrained divertor-chamber field grows a
# large psiN~1 lobe out to R~1.35 that deflects the X2-SE leg away from the
# E2 plate (INGRID: "target does not intersect field line").  Anchors are
# tied to the primary X-point flux and translated with their parent X-point
# when the targets move during a sweep.
# --------------------------------------------------------------------------
REF_XPT1 = (0.6733, -1.2624)
REF_XPT2 = (0.6812, -1.3219)
REF_ANCHORS_X1 = [(0.3813, -1.3513),   # W1 strike point
                  (0.5000, -1.3056),   # W1 leg, mid
                  (0.8744, -1.3254),   # E1 strike point
                  (1.0185, -1.4800)]   # outboard lobe extremity (index 3;
                                       # auto-dropped for outboard-primary SF,
                                       # see leg_anchor_isoflux)
REF_ANCHORS_X2 = [(0.5647, -1.5306),   # W2 strike point
                  (0.6287, -1.4000),   # W2 leg, mid
                  (0.8197, -1.5270),   # E2 strike point
                  (0.7293, -1.4000)]   # E2 leg, mid
# Private-flux depth anchors: REF psiN=0.963 (= INGRID's psi_pf_1) crossing
# on each W plate, tied to the outboard-midplane point on the same REF
# surface.  Without them the flux span along W1 falls 4e-4 short of
# psi_pf_1 and INGRID's A1_S trace misses the plate.
REF_PF_W1 = (0.4562, -1.4262)
REF_PF_W2 = (0.5158, -1.4820)
REF_PF_CORE = (1.3808, -0.0067)


def leg_anchor_isoflux(xpt_primary, xpt_secondary=None):
    """Isoflux constraints tying the reference leg geometry to the primary
    X-point flux, rigidly translated with each X-point's displacement.

    The reference (SF75) has its primary X-point INBOARD of the secondary, so
    the outboard-lobe anchor (REF_ANCHORS_X1[3], far down-and-out) sits in
    empty divertor and behaves.  For an outboard-primary snowflake (SF165
    type: the active null is the outboard one, xpt_primary[0] >
    xpt_secondary[0]) that anchor translates rigidly into the opposite leg
    region and pins a spurious psiN~1 island there, so it is auto-dropped.
    The discriminator leaves all inboard-primary cases (fidelity test, SF75)
    byte-identical.

    With ``xpt_secondary=None`` (single-null / LSN) only the PRIMARY anchors
    are used -- the secondary leg anchors (REF_ANCHORS_X2 / REF_PF_W2) pin
    strikes in the divertor for a second LOWER null that doesn't exist here,
    so they are omitted.  All four primary anchors are kept (the outboard-lobe
    drop is a snowflake-only concern)."""
    d1 = (xpt_primary[0] - REF_XPT1[0], xpt_primary[1] - REF_XPT1[1])
    Rxp, Zxp = xpt_primary
    if xpt_secondary is None:
        iso = [(Rxp, Zxp, r + d1[0], z + d1[1]) for r, z in REF_ANCHORS_X1]
        iso += [(REF_PF_W1[0] + d1[0], REF_PF_W1[1] + d1[1], *REF_PF_CORE)]
        return iso
    d2 = (xpt_secondary[0] - REF_XPT2[0], xpt_secondary[1] - REF_XPT2[1])
    anchors_x1 = (REF_ANCHORS_X1[:3] if xpt_primary[0] > xpt_secondary[0]
                  else REF_ANCHORS_X1)        # drop outboard lobe if primary outboard
    iso = [(Rxp, Zxp, r + d1[0], z + d1[1]) for r, z in anchors_x1]
    iso += [(Rxp, Zxp, r + d2[0], z + d2[1]) for r, z in REF_ANCHORS_X2]
    iso += [(REF_PF_W1[0] + d1[0], REF_PF_W1[1] + d1[1], *REF_PF_CORE),
            (REF_PF_W2[0] + d2[0], REF_PF_W2[1] + d2[1], *REF_PF_CORE)]
    return iso


# --------------------------------------------------------------------------
# Solve: place two lower X-points (snowflake)
# --------------------------------------------------------------------------
def solve_snowflake(d, xpt_primary, xpt_secondary=None, wall_file=DEFAULT_WALL,
                    psi_sign=-1.0, gamma=1e-8, blend=0.5, maxits=300,
                    px=1000.0, leg_anchors=True, steps=1, freeze_lower_d=False):
    """Solve a lower-divertor equilibrium.  With ``xpt_secondary`` given, places
    two lower X-points (snowflake); with ``xpt_secondary=None`` it solves a
    single lower X-point (LSN) -- Stage B then carries one X-point and only the
    primary leg anchors."""
    tok = build_machine(wall_file, px=px, freeze_lower_d=freeze_lower_d)
    # Domain matched to the reference eqdsk so the output header is identical
    eq = freegs.Equilibrium(tokamak=tok,
                            Rmin=d["rleft"], Rmax=d["rleft"] + d["rdim"],
                            Zmin=d["zmid"] - d["zdim"] / 2.0,
                            Zmax=d["zmid"] + d["zdim"] / 2.0,
                            nx=d["nw"], ny=d["nh"])
    profiles = make_profiles(d, psi_sign)

    # Pin the core LCFS to the experimental shape: sample the input boundary
    # EVENLY IN POLOIDAL ANGLE (so points don't cluster near the X-point) and
    # tie each to the primary null's flux.  Covering the upper half locks the
    # core vertically and stops the upward drift.
    def shape_iso_for(Rxp, Zxp):
        b = d["bdry"]
        keep = b[b[:, 1] > Zxp + 0.20]              # exclude divertor legs
        Rc, Zc = keep[:, 0].mean(), keep[:, 1].mean()
        ang = np.arctan2(keep[:, 1] - Zc, keep[:, 0] - Rc)
        order = np.argsort(ang); keep = keep[order]; ang = ang[order]
        targ = np.linspace(ang[0], ang[-1], 10)
        sel = [int(np.argmin(np.abs(ang - t))) for t in targ]
        return [(Rxp, Zxp, keep[i, 0], keep[i, 1]) for i in sorted(set(sel))]

    # --- Stage A: single lower X-point -> clean LSN -----------------------
    # Solve at the REFERENCE'S OWN primary null, not the requested one: with
    # a near-double-null reference (e.g. g049465.084044, upper X at
    # psiN~1.007) asking stage A for an X-point even ~mm off the natural null
    # throws Picard into a limit cycle, the boundary flipping between the
    # lower null and the upper X-point (Ip slamming 0.6<->1.0 MA, period ~5).
    # The natural null is the file's own equilibrium, so this always
    # converges; stage B then walks to the requested targets from there.
    # NB: do NOT anchor an outer leg at the legacy (1.20, -1.55) point: that
    # forces a large spurious psiN=1 lobe through the divertor chamber.
    nat = lower_xpoints(d)
    nat_prim = (min(nat, key=lambda p: np.hypot(p[0] - xpt_primary[0],
                                                p[1] - xpt_primary[1]))
                if nat else xpt_primary)
    nat_sec = None
    if xpt_secondary is not None:
        nat_sec = (min(nat, key=lambda p: np.hypot(p[0] - xpt_secondary[0],
                                                   p[1] - xpt_secondary[1]))
                   if len(nat) > 1 else xpt_secondary)
    Rxp, Zxp = nat_prim
    d1 = (Rxp - REF_XPT1[0], Zxp - REF_XPT1[1])
    iso_a = shape_iso_for(Rxp, Zxp) + [
        (Rxp, Zxp, REF_ANCHORS_X1[0][0] + d1[0], REF_ANCHORS_X1[0][1] + d1[1]),
        (Rxp, Zxp, REF_ANCHORS_X1[2][0] + d1[0], REF_ANCHORS_X1[2][1] + d1[1])]
    con_a = freegs.control.constrain(xpoints=[nat_prim], gamma=gamma,
                                     isoflux=iso_a)
    con_a(eq)
    print(f"  [stage A] LSN solve at reference's natural X-point "
          f"({Rxp:.4f}, {Zxp:+.4f}) ...")
    freegs.solve(eq, profiles, con_a, show=False, blend=blend, maxits=maxits)

    # --- Stage B: add the secondary X-point -> snowflake -----------------
    # core-shape isoflux pins the LCFS; leg anchors pin the divertor leg
    # geometry (strike points on the right plates).  No null-null flux tie:
    # tying the X-points to identical flux drives a degenerate closed pocket
    # (spurious divertor O-point).  With steps > 1 the targets are walked
    # from the natural nulls in equal increments (continuation), re-solving
    # at each — use for sweep members far (>~2 cm) from the reference nulls.
    if leg_anchors and xpt_secondary is not None and xpt_primary[0] > xpt_secondary[0]:
        print("  [stage B] outboard-primary SF (e.g. SF165): auto-dropping "
              "the outboard-lobe leg anchor")
    for k in range(1, steps + 1):
        f = k / steps
        prim_k = (nat_prim[0] + f * (xpt_primary[0] - nat_prim[0]),
                  nat_prim[1] + f * (xpt_primary[1] - nat_prim[1]))
        lbl = f" (step {k}/{steps})" if steps > 1 else ""
        if xpt_secondary is None:
            iso_b = shape_iso_for(*prim_k) + (leg_anchor_isoflux(prim_k)
                                              if leg_anchors else [])
            con_b = freegs.control.constrain(xpoints=[prim_k], gamma=gamma,
                                             isoflux=iso_b)
            print(f"  [stage B] single-null solve (one lower X-point){lbl} ...")
        else:
            sec_k = (nat_sec[0] + f * (xpt_secondary[0] - nat_sec[0]),
                     nat_sec[1] + f * (xpt_secondary[1] - nat_sec[1]))
            iso_b = shape_iso_for(*prim_k) + (leg_anchor_isoflux(prim_k, sec_k)
                                              if leg_anchors else [])
            con_b = freegs.control.constrain(xpoints=[prim_k, sec_k],
                                             gamma=gamma, isoflux=iso_b)
            print(f"  [stage B] snowflake solve (two lower X-points){lbl} ...")
        freegs.solve(eq, profiles, con_b, show=False, blend=blend,
                     maxits=maxits)
    return eq


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def null_census(R, Z, psi, psi_ax, psi_b, window=(0.3, 1.4, -1.75, -1.0),
                label="divertor"):
    """Census of ALL nulls (X and O) near the separatrix in `window`.
    A griddable snowflake should show ONLY the two intended X-points within
    psiN ~ [0.99, 1.05]; spurious O-points or extra X-points in that band
    are what break INGRID's leg tracing."""
    sp = RectBivariateSpline(R, Z, psi, kx=4, ky=4)
    r0, r1, z0, z1 = window
    Rf = np.linspace(r0, r1, 320); Zf = np.linspace(z0, z1, 320)
    g = np.hypot(sp(Rf, Zf, dx=1), sp(Rf, Zf, dy=1))
    mn = minimum_filter(g, size=7)
    out = []
    for ir, iz in np.argwhere((g == mn) & (g < np.percentile(g, 8))):
        rr, zz = Rf[ir], Zf[iz]
        for _ in range(60):
            gr = sp(rr, zz, dx=1)[0, 0]; gz = sp(rr, zz, dy=1)[0, 0]
            grr = sp(rr, zz, dx=2)[0, 0]; gzz = sp(rr, zz, dy=2)[0, 0]
            grz = sp(rr, zz, dx=1, dy=1)[0, 0]
            try:
                st = np.linalg.solve([[grr, grz], [grz, gzz]], [gr, gz])
            except np.linalg.LinAlgError:
                break
            rr -= st[0]; zz -= st[1]
            if not (r0 - 0.05 < rr < r1 + 0.05 and z0 - 0.05 < zz < z1 + 0.05):
                break
            if np.hypot(*st) < 1e-9:
                break
        if r0 < rr < r1 and z0 < zz < z1:
            grr = sp(rr, zz, dx=2)[0, 0]; gzz = sp(rr, zz, dy=2)[0, 0]
            grz = sp(rr, zz, dx=1, dy=1)[0, 0]
            typ = "X" if grr * gzz - grz ** 2 < 0 else "O"
            pn = (float(sp(rr, zz)[0, 0]) - psi_ax) / (psi_b - psi_ax)
            out.append((round(float(rr), 4), round(float(zz), 4),
                        round(pn, 4), typ))
    u = []
    for p in out:
        if not any(abs(p[0]-q[0]) < 1.5e-2 and abs(p[1]-q[1]) < 1.5e-2
                   for q in u):
            u.append(p)
    u.sort(key=lambda p: abs(p[2] - 1.0))
    print(f"  {label} null census (psiN, axis=0/primary=1):")
    for r, z, pn, t in u:
        flag = "  <-- near separatrix" if 0.985 < pn < 1.06 else ""
        print(f"    {t} R={r:.4f} Z={z:+.4f} psiN={pn:.4f}{flag}")
    return u


def divertor_audit(eq):
    R, Z, psi = eq.R[:, 0], eq.Z[0, :], eq.psi()
    opt, xpt = critical.find_critical(eq.R, eq.Z, psi)
    psi_ax = opt[0][2]
    psi_b = xpt[0][2] if xpt else psi_ax
    return null_census(R, Z, psi, psi_ax, psi_b)


def audit_file(fn, targets):
    """Audit the FINAL written G-EQDSK exactly as INGRID will see it."""
    d = parse_geqdsk(fn)
    print(f"\n  == final file audit: {fn} ==")
    low = lower_xpoints(d)
    print("  lower X-points  (file vs target):")
    for x in low:
        t = min(targets, key=lambda p: np.hypot(p[0]-x[0], p[1]-x[1]))
        err = 1e3 * np.hypot(t[0]-x[0], t[1]-x[1])
        print(f"    R={x[0]:.4f} Z={x[1]:+.4f}   "
              f"(target R={t[0]:.4f} Z={t[1]:+.4f}, miss {err:.1f} mm)")
    # final coordinates in INGRID's grid_settings keys, ready to paste into
    # the yaml (primary/secondary assigned by proximity to the targets)
    print("  INGRID yaml (grid_settings):")
    print(f"    rmagx: {d['rmaxis']:.4f}")
    print(f"    zmagx: {d['zmaxis']:.4f}")
    for t, sfx in zip(targets, ("", "2")):
        x = min(low, key=lambda p: np.hypot(p[0]-t[0], p[1]-t[1]))
        print(f"    rxpt{sfx}: {x[0]:.4f}")
        print(f"    zxpt{sfx}: {x[1]:.4f}")
    null_census(d["R"][1:-1], d["Z"][1:-1], d["psi"][1:-1, 1:-1],
                d["simag"], d["sibry"])
    null_census(d["R"][1:-1], d["Z"][1:-1], d["psi"][1:-1, 1:-1],
                d["simag"], d["sibry"], window=(0.27, 1.45, -0.2, 1.9),
                label="upper-half")
    return d


def report(eq, label, targets):
    opt, xpt = critical.find_critical(eq.R, eq.Z, eq.psi())
    print(f"\n  == {label} ==")
    print(f"  magnetic axis : R={opt[0][0]:.3f} Z={opt[0][1]:+.3f}")
    print(f"  plasma current: {eq.plasmaCurrent():.3e} A")
    psi_ax = opt[0][2]
    sib = xpt[0][2] if xpt else psi_ax
    dd = dict(R=eq.R[:, 0], Z=eq.Z[0, :], psi=eq.psi(), sibry=sib)
    low = lower_xpoints(dd)
    print("  lower X-points  (solved vs target):")
    for x in low:
        t = min(targets, key=lambda p: np.hypot(p[0]-x[0], p[1]-x[1]))
        err = 1e3 * np.hypot(t[0]-x[0], t[1]-x[1])
        print(f"    R={x[0]:.4f} Z={x[1]:+.4f}   "
              f"(target R={t[0]:.4f} Z={t[1]:+.4f}, miss {err:.1f} mm)")
    # upper region: the snowflake nulls must own the separatrix, so any
    # upper X-point should sit clearly outside (psiN > ~1.01)
    up = lower_xpoints(dd, window=(0.3, 1.2, 1.0, 1.8), ntop=2)
    if up:
        sp = RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], eq.psi(), kx=4, ky=4)
        print("  upper X-points (should be psiN > ~1.01):")
        for r, z in up:
            pn = (float(sp(r, z)[0, 0]) - psi_ax) / (sib - psi_ax)
            warn = "  <-- WARNING: competes with snowflake!" if pn < 1.005 else ""
            print(f"    R={r:.4f} Z={z:+.4f} psiN={pn:.4f}{warn}")
    divertor_audit(eq)
    return low


def plot_file(d, wall_file, out_png, title, targets):
    """Plot the final file's psiN map (what INGRID sees)."""
    pn = (d["psi"] - d["simag"]) / (d["sibry"] - d["simag"])
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.contour(d["R"], d["Z"], pn.T, levels=np.linspace(0.1, 0.9, 9),
               colors="C0", linewidths=0.6)
    ax.contour(d["R"], d["Z"], pn.T, levels=[1.0], colors="r", linewidths=1.4)
    ax.contour(d["R"], d["Z"], pn.T,
               levels=[1.02, 1.04, 1.06], colors="C1", linewidths=0.6)
    wall = np.loadtxt(wall_file, delimiter=",")
    ax.plot(wall[:, 0], wall[:, 1], "k-", lw=1)
    for r, z in targets:
        ax.plot(r, z, "kx", ms=8, mew=2)
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  saved {out_png}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MAST-U snowflake eqdsk factory: place the two lower "
                    "X-points and re-solve the free-boundary equilibrium.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("rxpt1", type=float, help="R of primary (active) X-point [m]")
    ap.add_argument("zxpt1", type=float, help="Z of primary (active) X-point [m]")
    ap.add_argument("rxpt2", type=float, nargs="?", default=None,
                    help="R of secondary X-point [m]; omit (with zxpt2) for a "
                         "single-null (LSN) equilibrium")
    ap.add_argument("zxpt2", type=float, nargs="?", default=None,
                    help="Z of secondary X-point [m]")
    ap.add_argument("--ref", default=str(DEFAULT_REF),
                    help="reference geqdsk supplying profiles + core shape")
    ap.add_argument("--wall", default=str(DEFAULT_WALL),
                    help="limiter file (R,Z comma-separated)")
    ap.add_argument("--psi-sign", type=float, default=-1.0,
                    help="sign matching file p'/ff' to FreeGS psi orientation")
    ap.add_argument("--gamma", type=float, default=1e-8,
                    help="Tikhonov coil-current regularisation; larger = "
                         "smoother/smaller currents, fewer spurious nulls")
    ap.add_argument("--maxits", type=int, default=300)
    ap.add_argument("--px", type=float, default=1000.0,
                    help="frozen Px circuit current [A]; calibrated so the "
                         "inner-leg region is free of spurious nulls")
    ap.add_argument("--no-leg-anchors", action="store_true",
                    help="disable the reference-derived divertor leg "
                         "isoflux anchors")
    ap.add_argument("--keep-divertor-coils", action="store_true",
                    help="[single-null] do NOT freeze the lower divertor coils. "
                         "By default LSN mode freezes D1L..D7L/DpL (which sculpt "
                         "the snowflake's second lower null); freezing them gives "
                         "a single lower X-point (the symmetric second null lands "
                         "up top and is removed by the reference-upper graft).")
    ap.add_argument("--steps", type=int, default=1,
                    help="number of continuation steps walking the X-points "
                         "from the reference's natural nulls to the targets; "
                         "increase for targets far (>~2 cm) from the nulls")
    ap.add_argument("--no-graft", action="store_true",
                    help="skip grafting the reference upper region (leaves "
                         "the physical upper X-point at psiN~1.002, which "
                         "sits inside INGRID's SOL band)")
    ap.add_argument("--graft-band", type=float, nargs=2, default=(0.1, 0.6),
                    metavar=("Z0", "Z1"),
                    help="Z blend window for the upper-region graft [m]")
    ap.add_argument("--out", default=None,
                    help="output geqdsk name (default auto from X-points)")
    args = ap.parse_args()

    if (args.rxpt2 is None) != (args.zxpt2 is None):
        ap.error("give BOTH rxpt2 and zxpt2 (snowflake) or NEITHER (single-null LSN)")
    prim = (args.rxpt1, args.zxpt1)
    sec = (args.rxpt2, args.zxpt2) if args.rxpt2 is not None else None
    lsn = sec is None
    targets = [prim] if lsn else [prim, sec]

    if args.out is None:
        args.out = (f"lsn_r1{prim[0]:.4f}_z1{prim[1]:+.4f}.geqdsk" if lsn else
                    f"sf_r1{prim[0]:.4f}_z1{prim[1]:+.4f}"
                    f"_r2{sec[0]:.4f}_z2{sec[1]:+.4f}.geqdsk")
    out_png = str(Path(args.out).with_suffix(".png"))

    d = parse_geqdsk(args.ref)
    print(f"reference: {args.ref}")
    print(f"  reference lower X-points: "
          f"{[(round(r, 4), round(z, 4)) for r, z in lower_xpoints(d)]}")
    print(f"  target primary  : ({prim[0]:.4f}, {prim[1]:+.4f})")
    if lsn:
        print("  single-null (LSN) mode: no secondary X-point")
    else:
        print(f"  target secondary: ({sec[0]:.4f}, {sec[1]:+.4f})")

    eq = solve_snowflake(d, prim, sec, wall_file=args.wall,
                         psi_sign=args.psi_sign, gamma=args.gamma,
                         maxits=args.maxits, px=args.px,
                         leg_anchors=not args.no_leg_anchors,
                         steps=args.steps,
                         freeze_lower_d=lsn and not args.keep_divertor_coils)
    report(eq, "solved equilibrium (pre-graft)", targets)

    with open(args.out, "w") as f:
        geqdsk.write(eq, f)

    out_d = parse_geqdsk(args.out)
    out_d = match_reference_signs(out_d, d)
    if not args.no_graft:
        out_d = graft_reference_upper(out_d, d, *args.graft_band)
        print(f"\n  grafted reference upper region "
              f"(blend Z in [{args.graft_band[0]}, {args.graft_band[1]}])")
    write_geqdsk_dict(out_d, args.out)

    final = audit_file(args.out, targets)
    title = (f"MAST-U LSN  X=({prim[0]:.3f},{prim[1]:.3f})" if lsn else
             f"MAST-U SF  X1=({prim[0]:.3f},{prim[1]:.3f}) "
             f"X2=({sec[0]:.3f},{sec[1]:.3f})")
    plot_file(final, args.wall, out_png, title, targets)
    print(f"  saved {args.out}")
