#!/usr/bin/env python3
"""
ExB_streamlines.py — streamlines of the ExB heat flux near the X-point(s), and
the poloidal profiles along the first SOL ring that explain them.

Figure 1  (qExB_streamlines.<fmt>, one panel per case, --ncols per row)
    Streamlines of the poloidal-plane ExB heat flux

        q_ExB = ( 5/2 (p_e + sum_i p_i) + sum_i 1/2 m_i n_i V_par,i^2 ) v_ExB

    drawn over the grey cell mesh in each case's colour from the heat-flux plots
    (SF_heat_transport_pp_xhermes_withfluxes.discover_cases), zoomed on the
    X-point region. (The slide writes the kinetic term as 1/2 m_i v_par^2; it
    is an energy density, so n_i is included here.)

    Line width ~ log10|q_ExB|: from --lw_min at the decade below the median |q|
    in the windows to --lw_max at the decade above its 99.5th percentile
    (--qrange), shared by all panels; the legend shows up to four round
    decades in between (--qref 1 10 gives the slide's). Weak flux is drawn
    thin, not dropped (see --qmin).

    Every line gets an arrow at its midpoint; longer lines get more, about
    every --arrow_spacing along them.

Figure 1b (qExB_phi_contours.<fmt>, same layout, colours and width scale)
    The same streamlines drawn without resampling or streamplot: v_E is
    perpendicular to grad(phi), so q_ExB streamlines are exactly phi contours.
    --nlevels contours, evenly spaced in phi within each window, are traced
    through the cell centres on the masked triangulation of the native mesh.
    Line spacing ~ 1/|v_E| (lines crowd where the E x B flow is fast), width =
    Hermes's |q_ExB|, arrows along v_E with the same rules as Figure 1.

Figure 2  (<case>_ring_profiles.<fmt>, one per case)
    Poloidal profiles along one flux tube (default: whichever first ring outside
    a separatrix has the longest path, i.e. the main SOL tube), walked target -> target through the real spatial
    neighbours across every branch cut:
        top     n_e (black) and T_e (red, twin axis)
        bottom  |E_theta| = |d phi / d l_pol|
    X-point crossings (branch cuts) are marked with dashed lines; a small R-Z
    panel shows which ring was used. With --ohm_terms an extra panel splits the
    signed E_theta with the poloidal projection of parallel Ohm's law,

        E_theta = (B/B_theta) eta J_par - (1/(e n_e)) grad_theta p_e
                  - 0.71/e grad_theta T_e

    plotting the two gradient terms and the resistive term as the remainder.

--------------------------------------------------------------------------------
Reuse
--------------------------------------------------------------------------------
Everything that already exists is imported by path, not rewritten:

  * hermes-drifts-0.py                    load_geometry (metric, B, corner mesh,
                                          true y-neighbours across cuts),
                                          compute_fields (phi -> E_r, E_pol,
                                          v_E,r, v_E,pol with the code's own
                                          metric and sign convention), _ddx,
                                          _ddy, _velocity, _mass,
                                          _use_aligned_open (per-rank time
                                          truncation for BOUT.dmp.* runs)
  * hermes-heat-pp-method-comparison.py   load_run, _tavg, _load_sf
  * Hermes-3_postprocessing.py            plot_separatrices (--seps)
  * SF_heat_transport_pp_xhermes_withfluxes.py   discover_cases

--------------------------------------------------------------------------------
v_ExB in (R, Z)
--------------------------------------------------------------------------------
Streamline DIRECTION (default --vE phi): in the poloidal plane
    v_E = s (B_t/B^2) phi_hat x grad(phi)
so the lines follow phi contours (closed contours = convective cells). phi is
interpolated with a C1 cubic over the cell centres and differentiated
analytically, which gives smooth lines. The sign s comes from a cell-by-cell
comparison with the metric version below (printed; it agrees in 100% of cells
on the MAST-U SF grids). Line WIDTH is always Hermes's own |q_ExB| from the
metric version. --vE metric uses the metric version for both (noisier lines).

Metric version, as Hermes evaluates it:
compute_fields gives the contravariant ExB velocities as Hermes evaluates them
(v^x = g11 g23/B^2 dphi/dy, v^y = -g11 g23/B^2 dphi/dx). The poloidal-plane
vector is then exact for any (also non-orthogonal) grid:

    v_(R,Z) = v^x d(R,Z)/dx + v^y d(R,Z)/dy

with d/dy taken across the true y-neighbours (so it is fine at the X-points).
The script prints |v_(R,Z)| / hypot(v_E,r, v_E,pol); it is 1 for an orthogonal
grid, so a large departure flags a metric/geometry mismatch.

The cell-centre vectors are interpolated linearly (Delaunay) onto a regular
R-Z grid for streamplot, and every point outside the cell mesh is masked, so no
streamline crosses the core hole or leaves the domain.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
    python ExB_streamlines.py <sims_dir> <grid_dir> [--only SF45 SF75 SF135] [options]

    # the slide: three snowflakes side by side, lower X-point region
    python ExB_streamlines.py CVODE_test grids/Hermes-3/sf-new-limiter \\
        --only SF45 SF75 SF135

    # fixed window, denser lines, ring 2 cells further out, Ohm's-law split
    python ExB_streamlines.py CVODE_test grids/Hermes-3/sf-new-limiter \\
        --only SF75 --zoom 0.3 1.05 -1.6 -0.6 --density 3 --ring 18 --ohm_terms
"""

import argparse
import glob
import importlib.util
import os
import sys
import tempfile
import traceback
import warnings

import dask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path


BANNER = "ExB_streamlines"

QE = 1.602176634e-19   # elementary charge [C]

_DRIFTS_CANDIDATES = [
    os.path.expanduser("~/Dev/My_working_scripts/Hermes-3_pp/hermes-drifts-0.py"),
]

# netCDF4/HDF5 is not thread-safe and xarray closes garbage-collected files
# without its lock; with dask's threaded scheduler that close races the next
# case's reads -> "NetCDF: HDF error" + segfault. Arrays are tiny: run serially.
dask.config.set(scheduler="synchronous")

# Branch cuts a ring can cross at an X-point. Which null each belongs to is not
# fixed for the INGRID snowflake grids, so crossings are labelled by cut name.
_CUTS = ("jyseps1_1", "jyseps2_1", "jyseps1_2", "jyseps2_2")


def _load_by_path(modname, candidates, explicit=None):
    """Import a toolchain script by file path (its filename is not an identifier)."""
    for path in ([explicit] if explicit else []) + candidates:
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"[{BANNER}] Loaded {modname} from {path}")
            return mod
    sys.exit(f"[{BANNER}] Could not find {modname} (looked in "
             f"{([explicit] if explicit else []) + candidates}).")


# ══════════════════════════════════════════════════════════════════════════════
# Physics: q_ExB on the cell centres
# ══════════════════════════════════════════════════════════════════════════════

def exb_heat_flux(run, geo, f, drifts, heat):
    """
    (qR, qZ, w) on the interior frame [W/m^2, W/m^2, J/m^3], or None if the run
    has no phi. w = 5/2 sum p + sum 1/2 m n V^2 is the convected energy density.
    """
    if f["phi"] is None or f["Pch"] is None:
        return None
    ds = run["ds"]
    w = 2.5 * f["Pch"]
    for ion in run["ions"]:
        Vi = drifts._velocity(ds, heat, run, ion)
        if Vi is None or f"N{ion}" not in ds.data_vars:
            print(f"[{BANNER}] {run['label']}: no V/N for {ion}; its kinetic "
                  f"energy is left out of q_ExB.")
            continue
        n = heat._tavg(ds[f"N{ion}"], run["t0"], run["t1"])
        w = w + 0.5 * drifts._mass(ds, heat, ion) * n * Vi ** 2

    R, Z, nbr = geo["R"], geo["Z"], geo["nbr"]
    vx = f["vEr"] * np.sqrt(geo["g11"])                  # contravariant v^x
    vy = f["vEpol"] / geo["hthe"]                        # contravariant v^y
    dRdx, dZdx = drifts._ddx(R, geo["dx"]), drifts._ddx(Z, geo["dx"])
    dRdy, dZdy = drifts._ddy(R, geo["dy"], nbr), drifts._ddy(Z, geo["dy"], nbr)
    vR = vx * dRdx + vy * dRdy
    vZ = vx * dZdx + vy * dZdy

    sx = slice(run["MXG"], run["nx"] - run["MXG"])

    # Same v_E from grad(phi) in (R, Z) (chain rule through d(R,Z)/d(x,y)):
    # v = s (B_t/B^2) phi_hat x grad(phi). The sign s and the direction
    # agreement fix how the smooth --vE phi streamlines are oriented.
    px, py = drifts._ddx(f["phi"], geo["dx"]), drifts._ddy(f["phi"], geo["dy"], nbr)
    with np.errstate(divide="ignore", invalid="ignore"):
        det = dRdx * dZdy - dZdx * dRdy
        pR = (dZdy * px - dZdx * py) / det
        pZ = (-dRdy * px + dRdx * py) / det
        F = geo["Bt"] / geo["B"] ** 2
        cos = ((F * pZ * vR - F * pR * vZ)
               / (np.hypot(F * pR, F * pZ) * np.hypot(vR, vZ)))[sx]
    cos = cos[np.isfinite(cos)]
    sign = 1.0 if np.median(cos) >= 0 else -1.0
    print(f"[{BANNER}] {run['label']}: v_E from phi contours vs metric, cell by cell: "
          f"sign {sign:+.0f}, {100 * np.mean(sign * cos > 0.97):.0f}% of cells within 14 deg")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.hypot(vR, vZ) / np.hypot(f["vEr"], f["vEpol"])
    q = np.hypot(w * vR, w * vZ)[sx]
    print(f"[{BANNER}] {run['label']}: |v_(R,Z)| / hypot(v_E,r, v_E,pol) median "
          f"{np.nanmedian(ratio[sx]):.3f} (1 for an orthogonal grid); "
          f"|q_ExB| p50/p99/max = " + " / ".join(
              f"{v / 1e6:.3g}" for v in np.nanpercentile(q, [50, 99, 100])) + " MW/m^2")
    return w * vR, w * vZ, w, sign


# ══════════════════════════════════════════════════════════════════════════════
# Mesh: polygons, window, raster mask
# ══════════════════════════════════════════════════════════════════════════════

def cell_polygons(geo, run):
    """
    (n_cells, 4, 2) corner polygons of the gridue cells that carry BOUT interior
    data (radial rows MXG-1 .. nx-MXG-2, every poloidal column including the
    target columns). Corners are sorted by angle about the centre, so the
    polygon is never a bow-tie whatever the corner numbering.
    """
    rm, zm = geo["rm"], geo["zm"]
    j0, j1 = run["MXG"] - 1, run["nx"] - run["MXG"] - 1
    r, z = rm[:, j0:j1, :], zm[:, j0:j1, :]
    r, z = r.reshape(-1, 5), z.reshape(-1, 5)
    ang = np.arctan2(z[:, 1:] - z[:, :1], r[:, 1:] - r[:, :1])
    order = np.argsort(ang, axis=1)
    rc = np.take_along_axis(r[:, 1:], order, axis=1)
    zc = np.take_along_axis(z[:, 1:], order, axis=1)
    poly = np.stack([rc, zc], axis=-1)
    good = np.all(np.isfinite(poly), axis=(1, 2)) & (np.ptp(rc, axis=1) + np.ptp(zc, axis=1) > 0)
    return poly[good]


def x_point(geo, run):
    """(R, Z) of the interior cell with the lowest B_pol, or None."""
    if geo["Bp"] is None:
        return None
    sx = slice(run["MXG"], run["nx"] - run["MXG"])
    k = np.unravel_index(np.nanargmin(np.abs(geo["Bp"][sx])), geo["Bp"][sx].shape)
    return geo["R"][sx][k], geo["Z"][sx][k]


def plot_window(geo, run, polys, args):
    """(Rmin, Rmax, Zmin, Zmax) for this case: --zoom, the X-point region, or all."""
    if args.zoom:
        return tuple(args.zoom)
    pr, pz = polys[..., 0], polys[..., 1]
    xp = x_point(geo, run) if args.region == "xpt" else None
    if xp is None:
        sel = np.ones(len(polys), bool)
    else:
        zc = pz.mean(axis=1)
        sel = zc < xp[1] + args.above if xp[1] <= 0 else zc > xp[1] - args.above
        print(f"[{BANNER}] {run['label']}: X-point (min B_pol) at R = {xp[0]:.3f}, "
              f"Z = {xp[1]:.3f} m")
    m = 0.02
    return (pr[sel].min() - m, pr[sel].max() + m, pz[sel].min() - m, pz[sel].max() + m)


def inside_mesh(polys, Rg, Zg):
    """Boolean mask of the regular-grid points that fall inside any cell."""
    r1, z1 = Rg[0, :], Zg[:, 0]
    mask = np.zeros(Rg.shape, bool)
    for p in polys:
        i0, i1 = np.searchsorted(r1, [p[:, 0].min(), p[:, 0].max()])
        k0, k1 = np.searchsorted(z1, [p[:, 1].min(), p[:, 1].max()])
        if i1 <= i0 or k1 <= k0:
            continue
        pts = np.column_stack([Rg[k0:k1, i0:i1].ravel(), Zg[k0:k1, i0:i1].ravel()])
        hit = Path(p).contains_points(pts, radius=1e-9).reshape(k1 - k0, i1 - i0)
        mask[k0:k1, i0:i1] |= hit
    return mask


def draw_mesh(ax, polys, color="0.55", lw=0.3):
    ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors=color,
                                     linewidths=lw, zorder=1))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — streamlines
# ══════════════════════════════════════════════════════════════════════════════

def _linewidth(q, args):
    """
    Line width for |q| [W/m^2]: linear in log10|q| from --lw_min at
    qrange[0] to --lw_max at qrange[1], clamped outside that range.
    """
    lo, hi = np.log10(np.asarray(args.qrange) * 1e6)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (np.log10(q) - lo) / (hi - lo)
    frac = np.clip(np.nan_to_num(frac, nan=0.0, neginf=0.0), 0.0, 1.0)
    return args.lw_min + frac * (args.lw_max - args.lw_min)


def _fmt_q(q_mw):
    """Legend label for a flux given in MW/m^2, with a sensible SI prefix."""
    for scale, unit in ((1.0, "MW"), (1e-3, "kW"), (1e-6, "W")):
        if q_mw >= scale or unit == "W":
            return rf"{q_mw / scale:g} {unit}m$^{{-2}}$"


def window_triangulation(case):
    """
    (tri, near, sx): Delaunay triangulation of the interior cell centres around
    the window (`near` selects them from the flattened interior arrays `[sx]`),
    with every triangle whose centroid lies outside the cell mesh masked, so
    nothing is interpolated or contoured across the core hole or off the grid.
    Cached on the case.
    """
    if "tri" in case:
        return case["tri"]
    run, geo, (qR, qZ, _, _) = case["run"], case["geo"], case["q"]
    R0, R1, Z0, Z1 = case["window"]
    sx = slice(run["MXG"], run["nx"] - run["MXG"])
    R, Z = geo["R"][sx].ravel(), geo["Z"][sx].ravel()
    pad = 0.15 * max(R1 - R0, Z1 - Z0)
    near = ((R > R0 - pad) & (R < R1 + pad) & (Z > Z0 - pad) & (Z < Z1 + pad)
            & np.isfinite(qR[sx].ravel()) & np.isfinite(qZ[sx].ravel()))
    tri = mtri.Triangulation(R[near], Z[near])
    cen = np.column_stack([tri.x[tri.triangles].mean(axis=1), tri.y[tri.triangles].mean(axis=1)])
    hit = np.zeros(len(cen), bool)
    for poly in case["polys"]:
        hit |= Path(poly).contains_points(cen)
    tri.set_mask(~hit)
    case["tri"] = (tri, near, sx)
    return case["tri"]


def interp_panel(case, args):
    """
    q_ExB components on a regular R-Z grid over the window, NaN off the mesh.

    --vE phi (default): phi is interpolated with a C1 cubic (matplotlib
    CubicTriInterpolator) over the cell centres and differentiated
    analytically, then
        v_(R,Z) = s (B_t / B^2) phi_hat x grad(phi),   s = +-1
    i.e. the streamlines follow phi contours, as E x B must. The sign s comes
    from a cell-by-cell comparison with the code-consistent metric version
    (exb_heat_flux). Only the direction is taken from phi: the magnitude
    (line width) is Hermes's own |q_ExB|, interpolated linearly. --vE metric interpolates that metric version
    (Hermes's own finite differences) linearly instead: faithful cell by cell,
    but its grid-scale noise makes the lines wiggle.

    Triangles whose centroid lies outside the cell mesh (e.g. across the core
    hole) are masked, so nothing is interpolated across them.
    """
    run, geo, (qR, qZ, w, sign) = case["run"], case["geo"], case["q"]
    R0, R1, Z0, Z1 = case["window"]

    # Regular grid over the window, args.res points along the longer side.
    h = max(R1 - R0, Z1 - Z0) / args.res
    Rg, Zg = np.meshgrid(np.arange(R0, R1 + h, h), np.arange(Z0, Z1 + h, h))

    tri, near, sx = window_triangulation(case)
    QR, QZ = qR[sx].ravel(), qZ[sx].ravel()

    def lin(values):
        return np.ma.filled(mtri.LinearTriInterpolator(tri, values)(Rg, Zg), np.nan)

    if args.vE == "phi":
        phi = case["f"]["phi"][sx].ravel()[near]
        fac = (geo["Bt"] / geo["B"] ** 2)[sx].ravel()[near]
        dphi_R, dphi_Z = (np.ma.filled(g, np.nan) for g in
                          mtri.CubicTriInterpolator(tri, phi, kind="geom").gradient(Rg, Zg))
        F = sign * lin(fac)
        uR, uZ = F * dphi_Z, -F * dphi_R
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = lin(np.hypot(QR, QZ)[near]) / np.hypot(uR, uZ)
        U, V = uR * scale, uZ * scale
    else:
        U, V = lin(QR[near]), lin(QZ[near])
    outside = ~inside_mesh(case["polys"], Rg, Zg)
    U[outside] = np.nan
    V[outside] = np.nan
    return Rg, Zg, U, V


def streamline_panel(ax, case, grid, args):
    geo, polys = case["geo"], case["polys"]
    R0, R1, Z0, Z1 = case["window"]
    Rg, Zg, U, V = (a.copy() for a in grid)
    qmag = np.hypot(U, V)
    if args.qmin > 0:
        weak = qmag < args.qmin * 1e6
        U[weak] = np.nan
        V[weak] = np.nan

    draw_mesh(ax, polys)
    if args.seps:
        case["pp"].plot_separatrices(ax, geo["rm"], geo["zm"], geo["grid"], geo["ny_inner"])

    # Broken (evenly spaced) lines by default: with the smooth phi-contour
    # field they come out long enough to show the cells; --unbroken lets them
    # run through each other. matplotlib's own arrows (one per line) are
    # switched off and placed along the lines instead.
    color = args.color or case["color"]
    kw = dict(density=args.density, color=color, linewidth=_linewidth(qmag, args),
              arrowstyle="-", minlength=args.minlength, maxlength=args.maxlength,
              broken_streamlines=args.broken, zorder=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = ax.streamplot(Rg, Zg, np.ma.masked_invalid(U), np.ma.masked_invalid(V), **kw)
    add_arrows(ax, sp.lines, args.arrow_spacing * np.hypot(R1 - R0, Z1 - Z0), color, args)

    _panel_decor(ax, case, color, args)


def add_arrows(ax, lines, spacing, color, args):
    """
    Arrows on a streamplot LineCollection. streamplot stores a line as
    consecutive 2-point segments in the flow direction, so a line is a run of
    segments where each one starts where the previous one ended.
    """
    segs = lines.get_segments()
    lws = np.broadcast_to(lines.get_linewidths(), (len(segs),))
    runs, cur = [], [0]
    for k in range(1, len(segs)):
        if np.allclose(segs[k][0], segs[k - 1][-1]):
            cur.append(k)
        else:
            runs.append(cur)
            cur = [k]
    if segs:
        runs.append(cur)
    place_arrows(ax, [(np.array([segs[r[0]][0]] + [segs[k][-1] for k in r]),
                       np.asarray(lws)[r]) for r in runs], spacing, color, args)


def place_arrows(ax, lines, spacing, color, args):
    """
    Arrow heads along each (points (n, 2) in flow order, segment widths (n-1,))
    line. Every line gets one arrow at its midpoint, always. Longer lines get
    more, about every `spacing` [m], but those extra ones are skipped when
    closer than spacing/2 to an arrow already drawn, so lines spiralling into a
    focus don't pile arrows into a blob.
    """
    placed = []
    for pts, lws in lines:
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        if len(pts) < 2 or s[-1] <= 0:
            continue
        n = max(1, int(s[-1] / spacing))
        targets = (np.arange(n) + 0.5) * s[-1] / n
        mid = n // 2                                       # the mandatory one
        for i in [mid] + [i for i in range(n) if i != mid]:
            k = min(max(int(np.searchsorted(s, targets[i])), 1), len(pts) - 1)
            if (i != mid and placed
                    and np.min(np.hypot(*(np.array(placed) - pts[k]).T)) < 0.5 * spacing):
                continue
            placed.append(pts[k])
            lw = float(lws[k - 1])
            ax.add_patch(FancyArrowPatch(
                pts[k - 1], pts[k], arrowstyle="-|>", color=color, lw=0.5 * lw,
                mutation_scale=10 * args.arrowsize * (0.7 + 0.3 * lw), zorder=4))


def _panel_grid(n, args):
    ncols = min(args.ncols, n)
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(args.panel_size[0] * ncols, args.panel_size[1] * nrows),
                            sharex=args.zoom is not None, sharey=args.zoom is not None)
    return fig, axs, ncols


def _finish_grid(fig, axs, n, ncols, title, name, args):
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    for k, ax in enumerate(axs.flat[:n]):
        if k % ncols == 0:
            ax.set_ylabel("Z [m]")
        if k + ncols >= n:                   # nothing below it: bottom of its column
            ax.set_xlabel("R [m]")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save(fig, args, name)


def _panel_decor(ax, case, color, args):
    """Width legend, case label and window limits, shared by both figures."""
    handles = [Line2D([], [], color=color, lw=float(_linewidth(np.array(q * 1e6), args)),
                      label=_fmt_q(q)) for q in args.qref]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.85,
              handlelength=2.5)
    ax.text(0.03, 0.03, case["label"], transform=ax.transAxes, fontsize=11,
            ha="left", va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8))
    R0, R1, Z0, Z1 = case["window"]
    ax.set_xlim(R0, R1)
    ax.set_ylim(Z0, Z1)
    ax.set_aspect("equal")


def figure_streamlines(cases, args):
    n = len(cases)
    fig, axs, ncols = _panel_grid(n, args)
    grids = [interp_panel(case, args) for case in cases]

    # Width scale shared by all panels, so widths compare across cases: from
    # the decade below the median |q| in the windows (thinnest) to the decade
    # above the 99.5th percentile (thickest). Legend on round decades in that
    # range, at most four, always including the top one.
    qall = np.concatenate([np.hypot(U, V).ravel() for _, _, U, V in grids]) / 1e6
    qall = qall[np.isfinite(qall) & (qall > 0)]
    if args.qrange is None:
        lo, hi = (np.percentile(qall, [50, 99.5]) if qall.size else (0.1, 10.0))
        args.qrange = [10.0 ** np.floor(np.log10(lo)), 10.0 ** np.ceil(np.log10(hi))]
    if args.qref is None:
        decades = 10.0 ** np.arange(np.log10(args.qrange[0]), np.log10(args.qrange[1]) + 0.5)
        step = max(1, int(np.ceil((len(decades) - 1) / 3)))
        args.qref = list(decades[::-1][::step][::-1])
    print(f"[{BANNER}] |q_ExB| in the windows: p50/p99.5/max = "
          + " / ".join(_fmt_q(v).split("m$")[0] for v in
                       np.percentile(qall, [50, 99.5, 100])) + " m^-2;  widths "
          + f"{args.lw_min:g}-{args.lw_max:g} over "
          + " - ".join(_fmt_q(q).split("m$")[0] for q in args.qrange) + ";  legend "
          + ", ".join(_fmt_q(q).split("m$")[0] for q in args.qref)
          + (f";  lines dropped below {_fmt_q(args.qmin).split('m$')[0]} m^-2"
             if args.qmin > 0 else ""))

    for ax, case, grid in zip(axs.flat, cases, grids):
        streamline_panel(ax, case, grid, args)
    _finish_grid(fig, axs, n, ncols, r"$\vec{q}_E$", "qExB_streamlines", args)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1b — phi contours on the native mesh (no resampling, no streamplot)
# ══════════════════════════════════════════════════════════════════════════════

def native_triangulation(case):
    """
    Triangulation of the interior cell centres with the MESH's own
    connectivity: each quad (i, j), (i+1, j), (i+1, j'), (i, j') with
    j' = up[i, j] = up[i+1, j] (true y-neighbour, so branch cuts are crossed
    where the cells really connect) is split into two triangles.

    At each null the two rings either side of the separatrix go to different
    neighbours, so those quads don't exist: the cells meeting at the null
    leave a small polygonal hole. Each hole (quads grouped by shared cells) is
    filled by a fan around ONE added node at the centroid of its cell centres,
    with phi = their mean: the only interpolated value in the figure.
    Returns (triangulation, phi at its nodes). Cached.
    """
    if "native_tri" in case:
        return case["native_tri"]
    run, geo = case["run"], case["geo"]
    m, nx, ny = run["MXG"], run["nx"], run["ny"]
    up = geo["nbr"][0]
    node = lambda i, j: (i - m) * ny + j                    # noqa: E731  [sx].ravel() order
    tris, holes = [], []
    for i in range(m, nx - m - 1):
        for j in range(ny):
            u0, u1 = up[i, j], up[i + 1, j]
            if u0 < 0 or u1 < 0:
                continue
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, u1), node(i, u0)
            if u0 == u1:
                tris += [(a, b, c), (a, c, d)]
            else:
                holes.append({a, b, c, d})

    sx = slice(m, nx - m)
    x, y = list(geo["R"][sx].ravel()), list(geo["Z"][sx].ravel())
    phi = list(case["f"]["phi"][sx].ravel())
    groups = []                                     # merge hole quads sharing cells
    for h in holes:
        for g in [g for g in groups if g & h]:
            h |= g
            groups.remove(g)
        groups.append(h)
    for g in groups:
        ids = np.array(sorted(g))
        px, py = np.array(x)[ids], np.array(y)[ids]
        cx, cy = px.mean(), py.mean()
        ring = ids[np.argsort(np.arctan2(py - cy, px - cx))]
        k = len(x)
        x.append(cx)
        y.append(cy)
        phi.append(float(np.mean(np.array(phi)[ids])))
        tris += [(k, int(ring[n]), int(ring[(n + 1) % len(ring)])) for n in range(len(ring))]
    case["native_tri"] = (mtri.Triangulation(np.array(x), np.array(y), np.array(tris)),
                          np.array(phi))
    return case["native_tri"]


def contour_panel(ax, case, args):
    """
    E x B streamlines drawn as contours of phi through the cell centres.

    In axisymmetry v_E,pol = s (B_t/B^2) phi_hat x grad(phi) is perpendicular
    to grad(phi), so its streamlines (and q_ExB's, w > 0) ARE phi contours.
    Levels are evenly spaced in phi between the 1st and 99th percentile of phi
    in the window, so the line spacing ~ 1/|grad phi| ~ 1/|v_E|: lines crowd
    where the E x B flow is fast. Contours are piecewise linear on the mesh's
    own connectivity, with the hole around each null filled
    (native_triangulation). Widths are Hermes's |q_ExB| at
    each segment and each contour is oriented along v_E (majority vote of
    s F (dphi/dZ, -dphi/dR) over its segments, sign s from exb_heat_flux),
    both evaluated on the window's Delaunay triangulation.
    """
    run, geo, f = case["run"], case["geo"], case["f"]
    qR, qZ, _, sign = case["q"]
    tri, near, sx = window_triangulation(case)
    R0, R1, Z0, Z1 = case["window"]
    phi = f["phi"][sx].ravel()[near]
    inwin = (tri.x > R0) & (tri.x < R1) & (tri.y > Z0) & (tri.y < Z1)
    lo, hi = np.nanpercentile(phi[inwin], [1, 99])
    levels = np.linspace(lo, hi, args.nlevels + 2)[1:-1]

    ntri, nphi = native_triangulation(case)
    cs = ax.tricontour(ntri, nphi, levels=levels)
    pieces = [poly for path in cs.get_paths()
              for poly in path.to_polygons(closed_only=False) if len(poly) >= 2]
    cs.remove()

    qlin = mtri.LinearTriInterpolator(tri, np.hypot(qR, qZ)[sx].ravel()[near])
    flin = mtri.LinearTriInterpolator(tri, (geo["Bt"] / geo["B"] ** 2)[sx].ravel()[near])
    grad = mtri.LinearTriInterpolator(tri, phi).gradient
    color = args.color or case["color"]
    minlen = args.contour_minlength * np.hypot(R1 - R0, Z1 - Z0)

    lines, segs, widths = [], [], []
    for pts in pieces:
        if np.sum(np.hypot(*np.diff(pts, axis=0).T)) < minlen:
            continue
        mids = 0.5 * (pts[1:] + pts[:-1])
        gR, gZ = (np.ma.filled(g, np.nan) for g in grad(mids[:, 0], mids[:, 1]))
        F = sign * np.ma.filled(flin(mids[:, 0], mids[:, 1]), np.nan)
        t = np.diff(pts, axis=0)
        if np.nansum(np.sign(t[:, 0] * F * gZ - t[:, 1] * F * gR)) < 0:
            pts, mids = pts[::-1], mids[::-1]
        lw = _linewidth(np.ma.filled(qlin(mids[:, 0], mids[:, 1]), np.nan), args)
        lines.append((pts, lw))
        segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
        widths.append(lw)

    draw_mesh(ax, case["polys"])
    if args.seps:
        case["pp"].plot_separatrices(ax, geo["rm"], geo["zm"], geo["grid"], geo["ny_inner"])
    if segs:
        ax.add_collection(LineCollection(np.concatenate(segs), linewidths=np.concatenate(widths),
                                         colors=color, capstyle="round", zorder=3))
    place_arrows(ax, lines, args.arrow_spacing * np.hypot(R1 - R0, Z1 - Z0), color, args)
    _panel_decor(ax, case, color, args)
    print(f"[{BANNER}] {case['label']}: {args.nlevels} phi levels, "
          f"delta phi = {levels[1] - levels[0]:.3g} V, {len(lines)} contour pieces")


def figure_contours(cases, args):
    """Same layout, colours and width scale as figure_streamlines (run it first)."""
    n = len(cases)
    fig, axs, ncols = _panel_grid(n, args)
    for ax, case in zip(axs.flat, cases):
        contour_panel(ax, case, args)
    _finish_grid(fig, axs, n, ncols,
                 r"$\vec{q}_E$ streamlines as $\phi$ contours (native mesh)",
                 "qExB_phi_contours", args)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — poloidal profiles along one ring
# ══════════════════════════════════════════════════════════════════════════════

def ring_path(geo, ix):
    """
    y indices of ring ix in spatial order, following the true y-neighbours from
    the first cell with no ylow neighbour (a target). A closed ring starts at 0.
    """
    up, dn = geo["nbr"]
    starts = np.where(dn[ix] < 0)[0]
    j = int(starts[0]) if starts.size else 0
    path, seen = [], set()
    while j >= 0 and j not in seen:
        path.append(j)
        seen.add(j)
        j = int(up[ix, j])
    return np.array(path)


def _cut_names(geo):
    g = geo["grid"]
    return {int(g[k].values): k for k in _CUTS if k in g}


def ring_crossings(geo, path):
    """[(k, name)]: the path crosses branch cut `name` between path[k] and path[k+1]."""
    names = _cut_names(geo)
    out = []
    for k in range(len(path) - 1):
        a, b = int(path[k]), int(path[k + 1])
        hit = names.get(a) if a in names else names.get(b - 1)
        if hit is None and b != a + 1:
            hit = "cut"
        if hit is not None:
            out.append((k, hit))
    return out


def figure_ring(case, args):
    run, geo, f = case["run"], case["geo"], case["f"]
    drifts, heat = case["drifts"], case["heat"]
    if args.ring is not None:
        ix = args.ring
        if not run["MXG"] <= ix < run["nx"] - run["MXG"]:
            print(f"[{BANNER}] --ring {ix} is not an interior x index; ring plot skipped.")
            return
    else:
        # The main SOL tube (target -> around the core -> target) is the longest
        # of the first rings outside each separatrix; in a snowflake the other
        # one only links two legs.
        cand = [i for i in (geo["ix_sep"], geo["ix_sep2"]) if i is not None]
        ix = max(cand, key=lambda i: len(ring_path(geo, i)))
    path = ring_path(geo, ix)
    R, Z = geo["R"][ix, path], geo["Z"][ix, path]
    seg = np.hypot(np.diff(R), np.diff(Z))
    s0 = 0.5 * geo["hthe"][ix, path[0]] * geo["dy"][ix, path[0]]
    s = s0 + np.concatenate([[0.0], np.cumsum(seg)])
    cross = ring_crossings(geo, path)
    print(f"[{BANNER}] {case['label']}: ring x = {ix} (separatrices at "
          f"{[geo['ix_sep']] + ([geo['ix_sep2']] if geo['ix_sep2'] else [])}), "
          f"{len(path)} cells, {s[-1] + s0:.2f} m; crosses "
          + (", ".join(f"{n} at {0.5 * (s[k] + s[k + 1]):.2f} m" for k, n in cross)
             or "no branch cut"))

    ne, Te = f["Ne"][ix, path], f["Te"][ix, path]
    Ep = f["Epol"][ix, path] if f.get("Epol") is not None else np.full(len(path), np.nan)
    nrow = 3 if args.ohm_terms and f.get("Epol") is not None else 2
    fig = plt.figure(figsize=(11, 4.6 if nrow == 2 else 6.4))
    gs = GridSpec(nrow, 2, width_ratios=[1, 3.2], wspace=0.25, hspace=0.08, figure=fig)
    axm = fig.add_subplot(gs[:, 0])
    axs = [fig.add_subplot(gs[0, 1])]
    axs += [fig.add_subplot(gs[r, 1], sharex=axs[0]) for r in range(1, nrow)]

    # R-Z mesh with the ring
    draw_mesh(axm, case["all_polys"], lw=0.2)
    axm.plot(R, Z, color="k", lw=2.5)
    for k, _ in cross:
        axm.plot(0.5 * (R[k] + R[k + 1]), 0.5 * (Z[k] + Z[k + 1]), "o", ms=4,
                 mfc="none", mec="tab:orange")
    axm.autoscale_view()
    axm.set_aspect("equal")
    axm.set_xlabel("R [m]")
    axm.set_ylabel("Z [m]")

    ax = axs[0]
    ax.plot(s, ne, color="k", label=r"$n_e$")
    ax.set_ylabel(r"$n_e$ [m$^{-3}$]")
    axT = ax.twinx()
    axT.plot(s, Te, color="red", label=r"$T_e$")
    axT.set_ylabel(r"$T_e$ [eV]", color="red")
    axT.tick_params(axis="y", colors="red")
    ax.legend(handles=ax.get_lines() + axT.get_lines(), loc="upper center",
              fontsize=9, framealpha=0.85)
    ax.set_title(f"{case['label']}   ring x = {ix}   "
                 f"(t-avg of last {run['t1'] - run['t0'] + 1} outputs)", fontsize=10)

    ax = axs[1]
    ax.plot(s, np.abs(Ep), color="tab:blue")
    ax.set_ylabel(r"$|E_\theta|$ [Vm$^{-1}$]")

    if nrow == 3:
        ax = axs[2]
        dy, nbr, hthe = geo["dy"], geo["nbr"], geo["hthe"]
        Pe = heat._tavg(run["ds"]["Pe"], run["t0"], run["t1"]) if "Pe" in run["ds"] else None
        ax.plot(s, Ep, color="tab:blue", label=r"$E_\theta = -\nabla_\theta\phi$")
        if Pe is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                gp = (-drifts._ddy(Pe, dy, nbr) / hthe / (QE * f["Ne"]))[ix, path]
            gT = (-0.71 * drifts._ddy(f["Te"], dy, nbr) / hthe)[ix, path]
            ax.plot(s, gp, color="k", ls="--", lw=1,
                    label=r"$-\nabla_\theta p_e/(e n_e)$")
            ax.plot(s, gT, color="red", ls="--", lw=1, label=r"$-0.71\nabla_\theta T_e/e$")
            ax.plot(s, Ep - gp - gT, color="0.5", ls=":", lw=1.2,
                    label=r"$(B/B_\theta)\eta J_\parallel$ (remainder)")
        else:
            print(f"[{BANNER}] {case['label']}: no Pe in the dump; Ohm's-law terms skipped.")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_ylabel(r"$E_\theta$ terms [Vm$^{-1}$]")
        ax.legend(fontsize=8, ncol=2, framealpha=0.85)

    for a in axs:
        for k, name in cross:
            a.axvline(0.5 * (s[k] + s[k + 1]), color="0.35", ls="--", lw=1)
        if a is not axs[-1]:
            plt.setp(a.get_xticklabels(), visible=False)
    for n, (k, name) in enumerate(cross):              # staggered so close cuts stay legible
        axs[1].text(0.5 * (s[k] + s[k + 1]), 0.95 - 0.12 * (n % 3), f" X-pt ({name})",
                    transform=axs[1].get_xaxis_transform(), fontsize=8, color="0.35",
                    ha="left", va="top")
    axs[-1].set_xlabel("Poloidal distance [m]")
    _save(fig, args, f"{case['label']}_ring{ix}_profiles")


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════

def discover(sf, sims_dir, grid_dir):
    """
    discover_cases over every case in sims_dir. grid_dir may be flat (*.nc) or
    hold one subfolder per case (e.g. grids/Hermes-3/sf-new-limiter/SF45/...);
    the latter is flattened into a temporary folder of symlinks, and the grid
    paths are resolved back to the real files.
    """
    if glob.glob(os.path.join(grid_dir, "*.nc")):
        return sf.discover_cases(sims_dir, grid_dir)
    with tempfile.TemporaryDirectory() as flat:
        for g in sorted(glob.glob(os.path.join(grid_dir, "*", "*.nc"))):
            link = os.path.join(flat, os.path.basename(g))
            if not os.path.exists(link):
                os.symlink(os.path.abspath(g), link)
        cases = sf.discover_cases(sims_dir, flat)
        for c in cases:
            c["grid"] = os.path.realpath(c["grid"])
    return cases


def _save(fig, args, name):
    path = os.path.join(args.outdir, f"{name}.{args.fmt}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[{BANNER}] Saved {path}")
    if not args.show:
        plt.close(fig)


def load_case(c, drifts, heat, pp, args):
    run = heat.load_run(c["path"], c["grid"], twindow=args.twindow)
    run.update(case=c["label"], label=c["label"])
    geo = drifts.load_geometry(c["grid"], run)
    if geo["rm"] is None:
        print(f"[{BANNER}] {c['label']}: grid has no rm/zm corner mesh; skipped.")
        return None
    f = drifts.compute_fields(run, geo, heat)
    all_polys = cell_polygons(geo, dict(run, MXG=1, nx=run["nx"]))
    polys = cell_polygons(geo, run)
    case = dict(label=c["label"], color=c["color"], run=run, geo=geo, f=f, polys=polys,
                all_polys=all_polys, drifts=drifts, heat=heat, pp=pp)
    case["q"] = exb_heat_flux(run, geo, f, drifts, heat)
    case["window"] = plot_window(geo, run, polys, args)
    return case


def main():
    parser = argparse.ArgumentParser(
        description="ExB heat-flux streamlines near the X-point(s) and the n_e, T_e, "
                    "E_theta profiles along a SOL ring, for Hermes-3 drift runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("sims_dir", help="Folder containing the SF<angle>/SN simulation folders")
    parser.add_argument("grid_dir", help="Folder containing the grid files, flat or "
                                         "one subfolder per case")
    parser.add_argument("--only", nargs="+", default=None, metavar="CASE",
                        help="Only these sim folders, in panel order, e.g. --only SF45 SF75 SF135")
    parser.add_argument("--twindow", type=int, default=5,
                        help="Average over the last N outputs")
    parser.add_argument("--region", choices=("xpt", "full"), default="xpt",
                        help="Streamline window: X-point region (cells below Z_xpt + --above) "
                             "or the whole mesh")
    parser.add_argument("--above", type=float, default=0.35,
                        help="[m] how far past the X-point (min B_pol) the 'xpt' window reaches")
    parser.add_argument("--zoom", type=float, nargs=4, default=None,
                        metavar=("RMIN", "RMAX", "ZMIN", "ZMAX"),
                        help="Fixed window [m] for every panel (overrides --region)")
    parser.add_argument("--vE", choices=("phi", "metric"), default="phi",
                        help="Streamline field: smooth phi-contour E x B (C1 cubic phi) or "
                             "Hermes's metric finite differences (noisier)")
    parser.add_argument("--res", type=int, default=400,
                        help="Regular-grid points along the window's longer side")
    parser.add_argument("--density", type=float, default=0.8, help="streamplot density")
    parser.add_argument("--minlength", type=float, default=0.15,
                        help="streamplot minlength (fraction of the window)")
    parser.add_argument("--maxlength", type=float, default=3.0,
                        help="streamplot maxlength (fraction of the window); larger = "
                             "longer lines, but spirals into a focus wind on longer")
    parser.add_argument("--unbroken", dest="broken", action="store_false",
                        help="Let streamlines run through each other (longer, but crowded) "
                             "instead of matplotlib's evenly spaced broken ones")
    parser.add_argument("--arrowsize", type=float, default=1.0, help="Arrow head size")
    parser.add_argument("--arrow_spacing", type=float, default=0.15,
                        help="Distance between arrows along a line (fraction of the "
                             "window diagonal)")
    parser.add_argument("--color", default=None,
                        help="One streamline colour for every case (default: each case's "
                             "colour from the heat-flux plots)")
    parser.add_argument("--qrange", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="[MW/m^2] |q| mapped to --lw_min .. --lw_max, log-scaled "
                             "(default: decade below the median to decade above the "
                             "99.5th percentile in the windows; the slide is ~0.1 30)")
    parser.add_argument("--qref", type=float, nargs="+", default=None,
                        help="[MW/m^2] legend values (default: up to four round decades "
                             "in --qrange; the slide used 1 10)")
    parser.add_argument("--lw_min", type=float, default=0.4,
                        help="Line width at qrange[0] and below")
    parser.add_argument("--lw_max", type=float, default=4.5,
                        help="Line width at qrange[1] and above")
    parser.add_argument("--nlevels", type=int, default=30,
                        help="phi levels in the phi-contour figure (evenly spaced between "
                             "the 1st and 99th percentile of phi in each window)")
    parser.add_argument("--contour_minlength", type=float, default=0.03,
                        help="Drop phi-contour pieces shorter than this (fraction of "
                             "the window diagonal)")
    parser.add_argument("--qmin", type=float, default=0.0,
                        help="[MW/m^2] stop streamlines where |q| is below this "
                             "(0 keeps everything; weak flux is drawn thin)")
    parser.add_argument("--ncols", type=int, default=4,
                        help="Streamline panels per row (rows are added as needed)")
    parser.add_argument("--panel_size", type=float, nargs=2, default=[3.6, 4.6],
                        metavar=("W", "H"), help="Size of one streamline panel [in]")
    parser.add_argument("--seps", action="store_true",
                        help="Overlay the separatrices on the streamline panels")
    parser.add_argument("--ring", type=int, default=None,
                        help="x index of the profile ring (default: first cell outside "
                             "the first separatrix)")
    parser.add_argument("--ohm_terms", action="store_true",
                        help="Add a panel splitting E_theta into Ohm's-law terms")
    parser.add_argument("--outdir",
                        default="/Users/sruiz/Dev/PhD/SF_Sims/Modified_SF_Runs/"
                                "Post_Processing/Drifts/ExB_streamlines",
                        help="Output folder")
    parser.add_argument("--fmt", default="png", help="Figure format")
    parser.add_argument("--show", action="store_true", help="Also show the figures")
    parser.add_argument("--drifts_path", default=None, help="Explicit path to hermes-drifts-0.py")
    parser.add_argument("--pp_path", default=None, help="Explicit path to Hermes-3_postprocessing.py")
    parser.add_argument("--heat_path", default=None,
                        help="Explicit path to hermes-heat-pp-method-comparison.py")
    parser.add_argument("--sf_path", default=None,
                        help="Explicit path to SF_heat_transport_pp_xhermes_withfluxes.py")
    args = parser.parse_args()

    for d in (args.sims_dir, args.grid_dir):
        if not os.path.isdir(d):
            sys.exit(f"Not a directory: {d}")
    os.makedirs(args.outdir, exist_ok=True)

    drifts = _load_by_path("hermes_drifts", _DRIFTS_CANDIDATES, args.drifts_path)
    pp = drifts._load_module("hermes3_pp", drifts._PP_CANDIDATES, args.pp_path)
    heat = drifts._load_module("hermes_heat_pp", drifts._HEAT_CANDIDATES, args.heat_path)
    drifts._use_aligned_open(heat, pp)          # per-rank t truncation for BOUT.dmp.* runs
    sf = heat._load_sf(args.sf_path)

    # Discover EVERY case so each one gets the colour the heat-flux plots give
    # it (discover_cases colours by position in the full list); then filter.
    found = discover(sf, args.sims_dir, args.grid_dir)
    if args.only:                               # panels in the order asked for
        rank = {o.upper(): k for k, o in enumerate(args.only)}
        missing = set(rank) - {c["label"].upper() for c in found}
        if missing:
            print(f"[{BANNER}] --only: not found: {sorted(missing)}")
        found = sorted((c for c in found if c["label"].upper() in rank),
                       key=lambda c: rank[c["label"].upper()])
    if not found:
        sys.exit("No simulations found.")

    cases = []
    for c in found:
        print(f"\n[{BANNER}] ══════ {c['label']}")
        try:
            case = load_case(c, drifts, heat, pp, args)
        except Exception:
            print(f"[{BANNER}] {c['label']} failed to load:")
            traceback.print_exc()
            continue
        if case is None:
            continue
        try:
            figure_ring(case, args)
        except Exception:
            print(f"[{BANNER}] {c['label']}: ring profiles failed:")
            traceback.print_exc()
        if case["q"] is None:
            print(f"[{BANNER}] {c['label']}: no phi in the dump (no-drift run?); "
                  f"no streamlines.")
            continue
        cases.append(case)

    if cases:
        figure_streamlines(cases, args)          # sets the shared width scale
        figure_contours(cases, args)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
