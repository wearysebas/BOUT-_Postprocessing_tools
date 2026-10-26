#!/usr/bin/env python3
"""
plot_curv_field.py — inspect the curvature field (bxcv) in an INGRID BOUT++ grid file.

Why this matters
----------------
Hermes-3's `vorticity` / `relax_potential` components build the diamagnetic
(curvature) drive from the grid variables bxcvx, bxcvy, bxcvz:

    Curlb_B = 2 * bxcv / B        (+ I*bxcvx added to z for shifted-metric grids)
    Div(J_dia) = Curlb_B . Grad(p)   summed over charged species

If bxcv is missing, BOUT++'s Mesh::get() silently fills it with 0 and only
logs a warning. The run then has NO curvature drive: no diamagnetic current,
no Pfirsch-Schlüter closure, and no ES X-point convection. So check it.

Grid format (INGRID only)
-------------------------
Cell geometry comes from the INGRID corner arrays, as in visualize_in_grid.py:

    rm, zm   shape (n_pol, n_rad, 5)   k=0 centre, k=1..4 corners

Which corner index k sits at which (low/high x, low/high y) position is
DETECTED from the geometry, not assumed: each corner's offset from the cell
centre is decomposed onto the local x and y centre-to-centre directions, and
a majority vote over interior cells is taken. The UEDGE gridue convention is
1=(x-,y-), 2=(x-,y+), 3=(x+,y-), 4=(x+,y+). The script stops if the vote is
not clean. Getting this wrong flips dr/dy and corrupts C wherever the
poloidal direction is vertical, while the polygons still look fine.

    BOUT (ix, iy) -> gridue (i_pol, j_rad):
        j_rad = ix - 1                          (ix = 2 .. nx-3)
        i_pol = iy + 1   if iy <  ny_inner
        i_pol = iy + 3   if iy >= ny_inner

- The outermost two radial rows on each side (ix = 0, 1, nx-2, nx-1) are
  extrapolated guard cells with no stored corners. They are not drawn.
- The two gridue columns at the ny_inner cut (i_pol = ny_inner+1, ny_inner+2)
  have real geometry but no BOUT value. By default they are filled with the
  value of their BOUT neighbour (iy = ny_inner-1 and iy = ny_inner,
  respectively) so the plots have no hole. Use --no-fill-cut-guards to leave
  them blank.
- Fields are drawn on the corner mesh, so there are no gaps at branch cuts or
  X-points.
  - Default: cell values are averaged onto the shared corners (matched by
    position, so the averaging also works across branch cuts) and
    Gouraud-shaded, which gives smooth colour gradients.
  - --flat draws each cell in one flat colour, showing the raw per-cell values.
- The basis vectors dr/dx, dr/dy are computed per cell from its own face
  midpoints, so no difference ever crosses a branch cut.

What the script does
--------------------
1. Reports which curvature-related variables exist, and flags missing,
   all-zero or NaN components. Also checks that the rm/zm cell centres match
   Rxy/Zxy, i.e. that the index mapping is right.
2. Reconstructs the poloidal-plane vector of bxcv in (R, Z):
       V_pol = bxcvx * dr/dx + bxcvy * dr/dy
   For an n=0 (axisymmetric) field p(x, y), V_pol . Grad(p) is exactly the
   drive Hermes-3 uses.
3. Leading-order sanity check. bxcv is expected as (B/2) curl(b/B), which is
   the convention Hermes-3 assumes. Near the nulls B is ~toroidal, so
   bxcv_perp ~ (b x kappa + b x GradB/B)/2 ~ +/- Zhat / R. The check
   quantity is
       C = R * V_Z / ref,   ref = (|b_phi|^3 + |b_phi|)/2,   b_phi = Bt/B
   It should be ~ +/-1 with a uniform sign over the grid, and |R V_R| should
   be << |R V_Z| near the X-points. Off-null deviations of O(Bp^2/B^2)
   are physical shaping terms (larger in a spherical tokamak such as MAST-U).
   sign(C) is the Z direction of the ion grad-B drift (b x GradB):
   C < 0 means the ion grad-B drift points DOWN.
4. Plots the raw components, |R V_pol| with direction arrows, and C, plus a
   zoom on the low-Bp (null) region. That zoom is where an SF lives.

Usage
-----
    python plot_curv_field.py GRID.grd.nc
    python plot_curv_field.py GRID.grd.nc --zoom 0.7 -1.35 0.12   # R0 Z0 half-width [m]
    python plot_curv_field.py GRID.grd.nc --nullfrac 0.03 --edges --show
    python plot_curv_field.py GRID.grd.nc --flat       # raw per-cell colours
    python plot_curv_field.py GRID.grd.nc --out figs/sf75p

Outputs: <out>_curv_overview.png, <out>_curv_null.png. Text report on stdout.
Requires: numpy, matplotlib, netCDF4.
"""

import argparse
import os
import sys

import numpy as np


K_C = 0  # INGRID cell-centre index; corner indices 1..4 are detected (IngridCells.detect_orientation)


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_grid(path):
    try:
        import netCDF4
    except ImportError:
        sys.exit("netCDF4 not found: pip install netCDF4")

    out, strings = {}, {}
    with netCDF4.Dataset(path) as ds:
        for name, var in ds.variables.items():
            kind = getattr(var.dtype, "kind", "S")  # str-typed vars have dtype == str
            if kind not in "fiu":
                if var.size == 1:
                    try:
                        strings[name] = str(var[...].item() if hasattr(var[...], "item") else var[...])[:80]
                    except Exception:
                        pass
                continue
            a = var[...]
            if np.ma.isMaskedArray(a):
                a = a.astype(float).filled(np.nan)
            out[name] = np.asarray(a)
        out["_attrs"] = {k: ds.getncattr(k) for k in ds.ncattrs()}
    out["_strings"] = strings
    return out


def scalar(g, name, default=None):
    if name not in g:
        return default
    return int(np.asarray(g[name]).item())


def y_segments(g, ny):
    """Split [0, ny) at the branch cuts (used for drawing separatrix polylines)."""
    j11 = scalar(g, "jyseps1_1", -1)
    j21 = scalar(g, "jyseps2_1", ny // 2)
    j12 = scalar(g, "jyseps1_2", ny // 2)
    j22 = scalar(g, "jyseps2_2", ny - 1)
    nyi = scalar(g, "ny_inner", j21)
    if j21 == j12:  # single null: no upper X-point, core is continuous across ny_inner
        cuts = {j11 + 1, j22 + 1}
    else:          # double null (connected or disconnected), incl. SF-as-DN meshes
        cuts = {j11 + 1, j21 + 1, nyi, j12 + 1, j22 + 1}
    cuts = sorted(c for c in cuts if 0 < c < ny)
    edges = [0] + cuts + [ny]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


# ----------------------------------------------------------------------------
# INGRID cell geometry
# ----------------------------------------------------------------------------
class IngridCells:
    """Cell corners from INGRID rm/zm, mapped onto BOUT (ix, iy) indexing."""

    def __init__(self, g, nx, ny, segs):
        if "rm" not in g or "zm" not in g:
            sys.exit("  ! rm/zm corner arrays not found: this script expects an INGRID BOUT++ grid.")
        self.rm, self.zm = g["rm"], g["zm"]
        n_pol, n_rad = self.rm.shape[:2]
        self.nx, self.ny = nx, ny
        self.ny_inner = scalar(g, "ny_inner")
        if self.ny_inner is None:
            sys.exit("  ! ny_inner missing: cannot map BOUT y-index to gridue columns.")
        if n_pol < ny + 4 or n_rad < nx - 2:
            sys.exit(f"  ! rm/zm shape {self.rm.shape} inconsistent with nx={nx}, ny={ny} "
                     f"(expected n_pol >= ny+4, n_rad >= nx-2)")
        self.n_pol = n_pol

        iy = np.arange(ny)
        self.ipol = np.where(iy < self.ny_inner, iy + 1, iy + 3)   # (ny,)
        self.ix_valid = np.arange(2, nx - 2)
        jrad = self.ix_valid - 1

        # Corners on the BOUT index grid: (nx, ny, 5); NaN on x-guard rows
        self.cR = np.full((nx, ny, 5), np.nan)
        self.cZ = np.full((nx, ny, 5), np.nan)
        self.cR[2:nx - 2] = self.rm[self.ipol[None, :], jrad[:, None], :]
        self.cZ[2:nx - 2] = self.zm[self.ipol[None, :], jrad[:, None], :]

        self.k, self.k_agreement = self.detect_orientation(segs)
        k = self.k
        self.poly_order = [k["SW"], k["SE"], k["NE"], k["NW"]]  # closed cycle

    def detect_orientation(self, segs):
        """Find which corner index k is at (x-,y-), (x-,y+), (x+,y-), (x+,y+).

        Primary test is topological and exact: a high-x corner of cell
        (ix, iy) coincides with a corner of cell (ix+1, iy), a low-x corner
        with one of (ix-1, iy), and likewise in y. Only interior cells are
        used, never across a branch cut.

        Fallback, if the corners are not shared: classify y by the sign of
        the corner offset along the local y direction, then split each y-pair
        by which corner lies further along x. This compares the two corners
        of a face, so the flux-surface curvature cancels.
        """
        nx = self.nx
        cR, cZ = self.cR, self.cZ
        hits = {k: {"x+": 0, "x-": 0, "y+": 0, "y-": 0} for k in (1, 2, 3, 4)}
        geo = {k: [] for k in (1, 2, 3, 4)}  # fallback: (beta, alpha) per cell
        ncell = 0
        for a, b in segs:
            if b - a < 3 or nx < 8:
                continue
            I, Ip, Im = slice(3, nx - 3), slice(4, nx - 2), slice(2, nx - 4)
            J, Jp, Jm = slice(a + 1, b - 1), slice(a + 2, b), slice(a, b - 2)
            R0, Z0 = cR[I, J, K_C], cZ[I, J, K_C]
            txR, txZ = 0.5 * (cR[Ip, J, K_C] - cR[Im, J, K_C]), 0.5 * (cZ[Ip, J, K_C] - cZ[Im, J, K_C])
            tyR, tyZ = 0.5 * (cR[I, Jp, K_C] - cR[I, Jm, K_C]), 0.5 * (cZ[I, Jp, K_C] - cZ[I, Jm, K_C])
            tol = 1e-3 * np.minimum(np.hypot(txR, txZ), np.hypot(tyR, tyZ))
            ok = np.isfinite(tol) & (tol > 0)
            ncell += int(ok.sum())
            nb = {"x+": (Ip, J), "x-": (Im, J), "y+": (I, Jp), "y-": (I, Jm)}
            for k in (1, 2, 3, 4):
                pR, pZ = cR[I, J, k], cZ[I, J, k]
                for key, (si, sj) in nb.items():
                    dist = np.hypot(cR[si, sj, 1:5] - pR[..., None], cZ[si, sj, 1:5] - pZ[..., None]).min(axis=-1)
                    hits[k][key] += int(np.count_nonzero((dist < tol) & ok))
                with np.errstate(divide="ignore", invalid="ignore"):
                    tyy = txR * txR + txZ * txZ, tyR * tyR + tyZ * tyZ
                    beta = ((pR - R0) * tyR + (pZ - Z0) * tyZ) / tyy[1]
                    alpha = ((pR - R0) * txR + (pZ - Z0) * txZ) / tyy[0]
                geo[k].append(np.stack([beta[ok], alpha[ok]], axis=1))
        if ncell == 0:
            sys.exit("  ! not enough interior cells to detect the rm/zm corner orientation")

        names = {(-1, -1): "SW", (-1, 1): "NW", (1, -1): "SE", (1, 1): "NE"}
        kmap, agree = {}, []
        for k in (1, 2, 3, 4):
            h = hits[k]
            sx = 1 if h["x+"] > h["x-"] else -1
            sy = 1 if h["y+"] > h["y-"] else -1
            agree.append(min(max(h["x+"], h["x-"]), max(h["y+"], h["y-"])) / ncell)
            kmap[names[(sx, sy)]] = k
        if len(kmap) == 4 and min(agree) > 0.9:
            return kmap, min(agree)

        # fallback: geometric, per-face comparison
        print("   (corners not shared between neighbouring cells; using geometric orientation fallback)")
        G = {k: np.concatenate(geo[k]) for k in (1, 2, 3, 4)}
        ysign = {k: np.sign(np.nanmedian(G[k][:, 0])) for k in G}
        hi = [k for k in G if ysign[k] > 0]
        lo = [k for k in G if ysign[k] < 0]
        if len(hi) != 2 or len(lo) != 2:
            sys.exit(f"  ! could not detect rm/zm corner orientation (y-signs {ysign})")
        kmap = {}
        votes = []
        for pair, (nlo, nhi) in ((lo, ("SW", "SE")), (hi, ("NW", "NE"))):
            k1, k2 = pair
            frac = np.mean(G[k2][:, 1] > G[k1][:, 1])  # is k2 further along +x than k1?
            votes.append(max(frac, 1 - frac))
            kx_hi, kx_lo = (k2, k1) if frac > 0.5 else (k1, k2)
            kmap[nlo], kmap[nhi] = kx_lo, kx_hi
        if min(votes) < 0.9:
            sys.exit(f"  ! ambiguous rm/zm corner orientation {kmap} (agreement {min(votes):.2f})")
        return kmap, min(votes)

    def ipol_of(self, iy):
        return iy + 1 if iy < self.ny_inner else iy + 3

    def basis(self, dx, dy):
        """Poloidal-plane covariant basis vectors from each cell's own face midpoints.

        e_x = (east face mid - west face mid) / dx
        e_y = (north face mid - south face mid) / dy
        """
        R, Z = self.cR, self.cZ
        sw, se, ne, nw = (self.k[n] for n in ("SW", "SE", "NE", "NW"))
        exR = (0.5 * (R[..., se] + R[..., ne]) - 0.5 * (R[..., sw] + R[..., nw])) / dx
        exZ = (0.5 * (Z[..., se] + Z[..., ne]) - 0.5 * (Z[..., sw] + Z[..., nw])) / dx
        eyR = (0.5 * (R[..., nw] + R[..., ne]) - 0.5 * (R[..., sw] + R[..., se])) / dy
        eyZ = (0.5 * (Z[..., nw] + Z[..., ne]) - 0.5 * (Z[..., sw] + Z[..., se])) / dy
        return exR, exZ, eyR, eyZ

    def polygons(self, fill_cut_guards=True):
        """Return (verts (n,4,2) ordered SW,SE,NE,NW; src_ix (n,); src_iy (n,)) for drawable cells."""
        PO = self.poly_order
        verts, sx, sy = [], [], []
        ixs, iys = np.meshgrid(self.ix_valid, np.arange(self.ny), indexing="ij")
        v = np.stack([self.cR[ixs, iys][..., PO],
                      self.cZ[ixs, iys][..., PO]], axis=-1)  # (nxv, ny, 4, 2)
        verts.append(v.reshape(-1, 4, 2))
        sx.append(ixs.ravel()); sy.append(iys.ravel())

        if fill_cut_guards:
            # gridue columns ny_inner+1 / ny_inner+2: real cells, no BOUT value.
            # Fill with the adjacent BOUT cell (iy = ny_inner-1 / ny_inner).
            for col, src_iy in ((self.ny_inner + 1, self.ny_inner - 1),
                                (self.ny_inner + 2, self.ny_inner)):
                if not (0 <= src_iy < self.ny):
                    continue
                jr = self.ix_valid - 1
                vg = np.stack([self.rm[col, jr][:, PO],
                               self.zm[col, jr][:, PO]], axis=-1)
                verts.append(vg)
                sx.append(self.ix_valid.copy()); sy.append(np.full(self.ix_valid.size, src_iy))
        verts = np.concatenate(verts)
        sx, sy = np.concatenate(sx), np.concatenate(sy)
        ok = np.all(np.isfinite(verts), axis=(1, 2))
        return verts[ok], sx[ok], sy[ok]

    def separatrix_lines(self, ixsep, segs):
        """Polylines along the low-x edge of cells ix = ixsep (the separatrix), one per y-segment."""
        if not (2 <= ixsep <= self.nx - 3):
            return []
        jr = ixsep - 1
        lines = []
        for a, b in segs:
            cols = list(range(self.ipol_of(a), self.ipol_of(b - 1) + 1))
            if b == self.ny_inner:  # carry on through the two cut columns
                cols += [self.ny_inner + 1, self.ny_inner + 2]
            cols = [c for c in cols if 0 <= c < self.n_pol]
            if not cols:
                continue
            sw, nw = self.k["SW"], self.k["NW"]
            Rl = [self.rm[c, jr, sw] for c in cols] + [self.rm[cols[-1], jr, nw]]
            Zl = [self.zm[c, jr, sw] for c in cols] + [self.zm[cols[-1], jr, nw]]
            lines.append((np.array(Rl), np.array(Zl)))
        return lines


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
class Painter:
    """Draws cell-valued fields on the INGRID corner mesh (no holes).

    Default: smooth. Cell values are averaged onto the shared corners, and
    each cell is split into 2 triangles and Gouraud-shaded. Corners are merged
    by position, so the averaging also works across branch cuts and X-points.
    flat=True: one colour per cell (the raw values).
    """

    def __init__(self, cells, segs, g, fill_cut_guards=True, edges=False, flat=False):
        self.verts, self.sx, self.sy = cells.polygons(fill_cut_guards)  # (n,4,2) SW,SE,NE,NW
        self.cells, self.segs, self.g = cells, segs, g
        self.edges, self.flat = edges, flat

        # shared corner nodes, merged by position
        pts = self.verts.reshape(-1, 2)
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-9)
        keys = np.round(pts / (1e-7 * span)).astype(np.int64)
        _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inv = np.asarray(inv).reshape(-1)
        xy = np.zeros((cnt.size, 2))
        np.add.at(xy, inv, pts)
        self.node_xy = xy / cnt[:, None]
        self.cell_nodes = inv.reshape(-1, 4)

    def field(self, ax, F, cmap=None, norm=None, clim=None):
        vals = F[self.sx, self.sy]
        ok = np.isfinite(vals)
        vmin, vmax = clim if clim is not None else (None, None)
        if self.flat:
            from matplotlib.collections import PolyCollection
            m = PolyCollection(self.verts[ok], array=vals[ok], cmap=cmap, norm=norm,
                               edgecolors="face", linewidths=0.2, rasterized=True)
            if clim is not None:
                m.set_clim(vmin, vmax)
            ax.add_collection(m)
            ax.autoscale_view()
        else:
            from matplotlib.tri import Triangulation
            cn = self.cell_nodes[ok]
            nn = len(self.node_xy)
            wsum = np.bincount(cn.ravel(), weights=np.repeat(vals[ok], 4), minlength=nn)
            wcnt = np.bincount(cn.ravel(), minlength=nn)
            nodal = np.where(wcnt > 0, wsum / np.maximum(wcnt, 1), 0.0)
            tri = np.concatenate([cn[:, [0, 1, 2]], cn[:, [0, 2, 3]]])
            triang = Triangulation(self.node_xy[:, 0], self.node_xy[:, 1], tri)
            m = ax.tripcolor(triang, nodal, shading="gouraud", cmap=cmap, norm=norm,
                             vmin=None if norm is not None else vmin,
                             vmax=None if norm is not None else vmax, rasterized=True)
        if self.edges:
            from matplotlib.collections import PolyCollection
            ax.add_collection(PolyCollection(self.verts, facecolors="none", edgecolors="k",
                                             linewidths=0.1, rasterized=True))
        return m

    def separatrices(self, ax):
        nx = self.cells.nx
        for key, ls in (("ixseps1", "-"), ("ixseps2", "--")):
            ix = scalar(self.g, key, nx)
            if ix is None or not (0 < ix < nx):
                continue
            for Rl, Zl in self.cells.separatrix_lines(ix, self.segs):
                ax.plot(Rl, Zl, color="0.15", lw=0.8, ls=ls)


def robust_sym(F, q=99.0):
    v = np.nanpercentile(np.abs(F), q)
    return v if np.isfinite(v) and v > 0 else 1.0


def quiver_unit(ax, R, Z, VR, VZ, n_across=22, mask=None):
    """Unit arrows of V_pol, thinned on a physical R-Z lattice.

    Grids are strongly non-uniform near X-points, so thinning by index stride
    either floods the null region or starves the core.
    """
    mag = np.hypot(VR, VZ)
    with np.errstate(invalid="ignore", divide="ignore"):
        uR, uZ = VR / mag, VZ / mag
    ok = np.isfinite(uR) & np.isfinite(uZ)
    if mask is not None:
        ok &= mask
    if not ok.any():
        return
    Rf, Zf, uRf, uZf = R[ok], Z[ok], uR[ok], uZ[ok]
    span = max(np.ptp(Rf), np.ptp(Zf), 1e-6)
    h = span / n_across
    keys = np.stack([np.floor(Rf / h), np.floor(Zf / h)], axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)  # one arrow per lattice bin
    ax.quiver(Rf[first], Zf[first], uRf[first], uZf[first], color="0.1",
              angles="xy", scale_units="xy", scale=1.6 / h, width=0.004,
              headwidth=3.5, headlength=4, headaxislength=3.5, pivot="mid")


def finish_axes(ax, title):
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title, fontsize=10)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grid", help="INGRID BOUT++ grid file (.nc / .grd.nc)")
    p.add_argument("--out", default=None, help="output prefix (default: grid file stem)")
    p.add_argument("--zoom", nargs=3, type=float, metavar=("R0", "Z0", "HW"),
                   help="zoom window centre and half-width [m] (default: auto from low-Bp cells)")
    p.add_argument("--nullfrac", type=float, default=0.05,
                   help="cells with Bp/B below this count as 'near null' (default 0.05)")
    p.add_argument("--arrows", type=int, default=22, help="approx. number of arrows across each panel (default 22)")
    p.add_argument("--flat", action="store_true",
                   help="one flat colour per cell (raw values) instead of smooth Gouraud shading")
    p.add_argument("--no-fill-cut-guards", action="store_true",
                   help="leave the two ny_inner cut columns blank instead of copying neighbour values")
    p.add_argument("--edges", action="store_true", help="draw thin cell edges")
    p.add_argument("--show", action="store_true", help="open interactive windows")
    args = p.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    g = load_grid(args.grid)
    out = args.out or os.path.splitext(os.path.splitext(args.grid)[0])[0]
    print(f"\n== Curvature check: {args.grid}")

    # ---- presence / sanity -------------------------------------------------
    for req in ("Rxy", "Zxy", "dx", "dy"):
        if req not in g:
            sys.exit(f"  ! required variable '{req}' missing from grid")

    comps = {c: g.get("bxcv" + c) for c in "xyz"}
    missing = [c for c, v in comps.items() if v is None]
    if missing:
        print("  !! MISSING: " + ", ".join("bxcv" + c for c in missing))
        print("     BOUT++ Mesh::get() will silently set these to 0. The Hermes-3 diamagnetic")
        print("     (curvature) drive will be wrong or absent. Add bxcv in the INGRID -> BOUT++")
        print("     conversion before trusting drift runs.")
        if "x" in missing or "y" in missing:
            sys.exit("     Cannot reconstruct the poloidal drive without bxcvx and bxcvy.")
    for c, v in comps.items():
        if v is None:
            continue
        nn = np.count_nonzero(~np.isfinite(v))
        nz = np.count_nonzero(v == 0.0)
        flag = "  !! ALL ZERO" if nz == v.size else ("  !! contains NaN/Inf" if nn else "")
        print(f"   bxcv{c}: min {np.nanmin(v): .3e}  max {np.nanmax(v): .3e}  "
              f"zeros {nz}/{v.size}{flag}")
    if all(comps[c] is not None and not np.any(comps[c]) for c in "xy"):
        print("  !! bxcvx and bxcvy are identically zero: Hermes-3 will run with NO n=0 curvature")
        print("     (diamagnetic) drive. Regenerate the curvature in the grid.")

    # ---- geometry ------------------------------------------------------------
    R, Z = g["Rxy"], g["Zxy"]
    nx, ny = R.shape
    if scalar(g, "ny", ny) != ny:
        sys.exit(f"  ! Rxy has {ny} y-columns but grid ny={scalar(g, 'ny')}: unexpected y guard cells")
    segs = y_segments(g, ny)
    cells = IngridCells(g, nx, ny, segs)
    print(f"   nx={nx} ny={ny}  rm/zm {cells.rm.shape}  y-segments: {segs}")
    k = cells.k
    conv = "UEDGE gridue convention" if (k["SW"], k["NW"], k["SE"], k["NE"]) == (1, 2, 3, 4) else "non-UEDGE order"
    print(f"   corner orientation (detected): (x-,y-)=k{k['SW']} (x-,y+)=k{k['NW']} "
          f"(x+,y-)=k{k['SE']} (x+,y+)=k{k['NE']}  [{conv}; agreement {100 * cells.k_agreement:.1f}%]")

    dev = np.nanmax(np.hypot(cells.cR[..., K_C] - R, cells.cZ[..., K_C] - Z))
    print(f"   rm/zm centre vs Rxy/Zxy: max offset {dev:.2e} m"
          + ("   !! > 1 mm: BOUT <-> gridue index mapping looks wrong" if dev > 1e-3 else ""))

    Vx, Vy = comps["x"], comps["y"]
    dx, dy = g["dx"], g["dy"]
    exR, exZ, eyR, eyZ = cells.basis(dx, dy)
    VR = Vx * exR + Vy * eyR
    VZ = Vx * exZ + Vy * eyZ
    CR, CZ = R * VR, R * VZ  # dimensionless; leading order ~ (0, +/-1)

    if "Btxy" in g and "Bxy" in g:
        bphi = np.abs(g["Btxy"] / g["Bxy"])
        ref = 0.5 * (bphi ** 3 + bphi)
    else:
        print("   (no Btxy/Bxy: reference factor set to 1)")
        ref = np.ones_like(R)
    C = CZ / ref  # NaN on x-guard rows (no corners)

    # near-null cells (drawable cells only)
    drawable = np.isfinite(C)
    if "Bpxy" in g and "Bxy" in g:
        bp_frac = g["Bpxy"] / g["Bxy"]
        near = (bp_frac < args.nullfrac) & drawable
        if near.sum() < 4:
            thr = np.nanpercentile(bp_frac[drawable], 2)
            near = (bp_frac <= thr) & drawable
            print(f"   no cells with Bp/B < {args.nullfrac}; using lowest 2% (Bp/B <= {thr:.3f})")
    else:
        bp_frac, near = None, None
        print("   (no Bpxy: near-null diagnostics skipped; use --zoom)")

    # ---- report --------------------------------------------------------------
    good = drawable & (np.abs(np.nan_to_num(C)) > 0.2)
    if good.any():
        maj = np.sign(np.nanmedian(C[good]))
        frac = np.mean(np.sign(C[good]) == maj)
        print(f"\n   Sign uniformity of C over grid: {100 * frac:.1f}% share sign {'+' if maj > 0 else '-'}"
              + ("   <-- expect ~100%; patches of opposite sign = broken curvature" if frac < 0.97 else ""))
        print(f"   => ion grad-B drift (b x GradB) points {'UP (+Z)' if maj > 0 else 'DOWN (-Z)'} in this grid."
              "  Check this against the experiment's field direction.")
    print(f"   C over whole grid: p5/p50/p95 = " + " / ".join(f"{q:+.3f}" for q in np.nanpercentile(C, [5, 50, 95])))
    if near is not None and near.any():
        cn = C[near]
        rat = np.abs(CR[near]) / np.maximum(np.abs(CZ[near]), 1e-30)
        print(f"   Near-null cells (n={near.sum()}): C p5/p50/p95 = "
              + " / ".join(f"{q:+.3f}" for q in np.nanpercentile(cn, [5, 50, 95]))
              + "   (expect ~ +/-1)")
        print(f"   Near-null |R V_R| / |R V_Z| median = {np.nanmedian(rat):.3f}   (expect << 1)")
        med = np.nanmedian(np.abs(cn))
        if not (0.7 < med < 1.3):
            print("   !! near-null |C| is far from 1: check the bxcv normalisation/units in the grid")
            print("      (Hermes-3 expects bxcv = (B/2) curl(b/B); e.g. stored as curl(b/B), or in")
            print("      normalised units, would show up here).")
    print("   Hermes-3 uses Curlb_B = 2*bxcv/B; n=0 drive = Curlb_B . Grad(p) uses only bxcvx, bxcvy.\n")

    paint = Painter(cells, segs, g, fill_cut_guards=not args.no_fill_cut_guards,
                    edges=args.edges, flat=args.flat)

    # ---- figure 1: overview --------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(14, 10), constrained_layout=True)
    for ax, (c, F) in zip(axs[0], (("x", Vx), ("y", Vy), ("z", comps["z"]))):
        if F is None:
            ax.text(0.5, 0.5, f"bxcv{c} missing", ha="center", va="center", transform=ax.transAxes)
            finish_axes(ax, f"bxcv{c}")
            continue
        v = robust_sym(F[drawable])
        m = paint.field(ax, F, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -v, v))
        paint.separatrices(ax)
        fig.colorbar(m, ax=ax, shrink=0.8, extend="both")
        finish_axes(ax, f"bxcv{c} (contravariant, raw)")

    ax = axs[1, 0]
    m = paint.field(ax, np.hypot(CR, CZ), cmap="viridis", clim=(0, 1.5))
    paint.separatrices(ax)
    quiver_unit(ax, R, Z, VR, VZ, n_across=args.arrows)
    fig.colorbar(m, ax=ax, shrink=0.8, extend="max")
    finish_axes(ax, r"$R\,|V_{pol}|$ + direction of $V_{pol}$ (expect ~1, vertical)")

    ax = axs[1, 1]
    m = paint.field(ax, C, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -1.5, 1.5))
    paint.separatrices(ax)
    fig.colorbar(m, ax=ax, shrink=0.8, extend="both")
    finish_axes(ax, r"$C = R V_Z/\mathrm{ref}$ (expect uniform $\pm1$)")

    ax = axs[1, 2]
    bins = np.linspace(-2, 2, 81)
    ax.hist(C[drawable].ravel(), bins=bins, color="0.55", label="all cells")
    if near is not None and near.any():
        ax.hist(C[near].ravel(), bins=bins, color="#c0392b", label=f"Bp/B < {args.nullfrac:g}")
    for xv in (-1, 1):
        ax.axvline(xv, color="0.2", lw=0.8, ls=":")
    ax.set_xlabel("C")
    ax.set_ylabel("cells")
    ax.legend(frameon=False)
    ax.set_title("Distribution of C", fontsize=10)
    fig.suptitle(os.path.basename(args.grid) + " — curvature field (bxcv)", fontsize=12)
    f1 = out + "_curv_overview.png"
    fig.savefig(f1, dpi=150)

    # ---- figure 2: zoom on null region -------------------------------------
    if args.zoom:
        R0, Z0, hw = args.zoom
    elif near is not None and near.any():
        R0, Z0 = np.nanmean(R[near]), np.nanmean(Z[near])
        hw = 1.6 * max(np.ptp(R[near]), np.ptp(Z[near]), 0.05) / 2 + 0.03
    else:
        R0 = Z0 = hw = None

    f2 = None
    if R0 is not None:
        win = (np.abs(R - R0) < hw) & (np.abs(Z - Z0) < hw)
        fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)

        ax = axs2[0]
        if bp_frac is not None:
            m = paint.field(ax, bp_frac, cmap="magma", clim=(0, max(3 * args.nullfrac, 0.1)))
            fig2.colorbar(m, ax=ax, shrink=0.85, extend="max")
            ttl = r"$B_p/B$ (null region)"
        else:
            ttl = "null region (no Bpxy)"
        paint.separatrices(ax)
        finish_axes(ax, ttl)

        ax = axs2[1]
        m = paint.field(ax, C, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -1.5, 1.5))
        paint.separatrices(ax)
        quiver_unit(ax, R, Z, VR, VZ, n_across=args.arrows, mask=win)
        fig2.colorbar(m, ax=ax, shrink=0.85, extend="both")
        finish_axes(ax, r"$C$ + direction of $V_{pol}$")

        ax = axs2[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ang = np.degrees(np.arctan2(np.abs(VR), np.abs(VZ)))  # 0 deg = vertical
        m = paint.field(ax, ang, cmap="viridis", clim=(0, 45))
        paint.separatrices(ax)
        fig2.colorbar(m, ax=ax, shrink=0.85, extend="max")
        finish_axes(ax, r"tilt of $V_{pol}$ from vertical [deg] (expect ~0 near nulls)")

        for ax in axs2:
            ax.set_xlim(R0 - hw, R0 + hw)
            ax.set_ylim(Z0 - hw, Z0 + hw)
        fig2.suptitle(f"Null region  (R0={R0:.3f}, Z0={Z0:.3f}, half-width {hw:.3f} m)", fontsize=12)
        f2 = out + "_curv_null.png"
        fig2.savefig(f2, dpi=150)

    print("   wrote:", f1 + ("" if f2 is None else ", " + f2))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
