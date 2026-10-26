"""
plot_psi.py — Plot poloidal flux (psixy) from a BOUT++ / Hermes-3 grid file,
as a cell heat map and/or contours, with zoom panels on the legs or X-points.

Supported grid formats
----------------------
  INGRID  : cell corners in rm/zm, shape (n_pol, n_rad, 5), k=0 centre,
            k=1 NW, k=2 SW, k=3 NE, k=4 SE. Polygon order NW→SW→SE→NE.
            n_pol = ny+4 (DN / snowflake, gap of 2 guard columns at ny_inner)
            n_pol = ny+2 (single null, no gap).
  Hypnotoad: cell corners in Rxy_corners / Rxy_lower_right_corners /
            Rxy_upper_right_corners / Rxy_upper_left_corners (+ Z), shape (nx, ny).

BOUT (ix, iy) ↔ gridue (i_pol, j_rad) mapping (INGRID)
  j_rad = ix - 1                                  (valid for ix = 2 .. nx-3)
  i_pol = iy + 1  if iy < ny_inner  else  iy + 3  (DN/SF, n_pol = ny+4)
  i_pol = iy + 1                                  (SN,    n_pol = ny+2)

Plot modes (--mode)
  heat     cells filled with psixy (no interpolation, exact cell values)
  contour  psi contours coloured by value
  both     heat map + thin black contours on top

Contours are drawn with tricontour on the cell centres. The triangulation is
a Delaunay one with every triangle whose centroid lies outside the gridded
domain masked, so contours never bridge the core hole, the region beyond the
targets, or the gap between legs. Psi is a continuous function of (R, Z), so
interpolating across branch cuts between physically adjacent cells is fine.

X-points are found from the grid geometry alone: a corner node shared by more
than 4 cells is an X-point (8 for a first-order null). Works for any topology.

Zoom panels (--zoom, one panel per token, side by side)
  full        whole grid
  legs        lower divertor: cells below the highest lower X-point
  upper_legs  upper divertor: cells above the lowest upper X-point
  xpoints     box enclosing all X-points found (+ --pad)
  xpt<i>      box of half-width --width around X-point i (e.g. xpt0, xpt1);
              X-points are numbered by increasing Z (printed at start-up)
  box         manual box, from --box RMIN RMAX ZMIN ZMAX

Separatrices / branch cuts (--plot_sep)
  INGRID DN/SF grids: cell-edge overlays, same convention as
  visualize_in_grid.py (branch cuts on corner 4/3, separatrices on corner 1).
  Other grids (INGRID SN, Hypnotoad): psi = psi_X contours, one per X-point,
  with psi_X the mean of the cells sharing the X-point node.

Usage
-----
    python plot_psi.py <grid_file> [--mode {heat,contour,both}] [--levels N]
                       [--zoom TOKEN [TOKEN ...]] [--xpt_width W] [--pad P]
                       [--box RMIN RMAX ZMIN ZMAX] [--cmap CMAP]
                       [--local_clim | --no-local_clim] [--no-edges]
                       [--plot_sep | --no-plot_sep] [--plot_xpt | --no-plot_xpt]
                       [--limiter PATH] [--save FILE]

Examples
--------
    python plot_psi.py mastu_sf75p_bout.grd.nc
    python plot_psi.py mastu_sf75p_bout.grd.nc --mode both --zoom full legs xpoints
    python plot_psi.py mastu_sf75p_bout.grd.nc --mode contour --levels 60 --zoom xpt0 xpt1
    python plot_psi.py mastu_sf45_bout.grd.nc --zoom box --box 0.5 0.9 -1.6 -1.1 --save psi.png
"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as mtri
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
import netCDF4


# Per-cut colour convention. Each branch cut gets its own colour; all dashed.
_BRANCH_CUT_COLORS = {
    "jyseps1_1": "gold",
    "jyseps2_1": "limegreen",
    "jyseps1_2": "cornflowerblue",
    "jyseps2_2": "tomato",
    "ny_inner": "cyan",
}

# Separatrix colours (ixseps1 magenta, ixseps2 yellow), as in visualize_in_grid.py
_SEP_COLORS = {"ixseps1": "magenta", "ixseps2": "yellow"}

# Colours for the psi = psi_X separatrix contours (fallback), one per X-point
_XPT_SEP_COLORS = ["magenta", "yellow", "orange", "cyan", "white"]

# Relative tolerance (fraction of the grid R extent) for merging corner nodes
_NODE_TOL_REL = 1e-6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_grid(path):
    """Return an open netCDF4 Dataset."""
    return netCDF4.Dataset(path)


def get_topology_indices(ds):
    """Extract all topology index scalars from the dataset."""
    return {
        "nx":        int(ds.variables["nx"][:]),
        "ny":        int(ds.variables["ny"][:]),
        "ixseps1":   int(ds.variables["ixseps1"][:]),
        "ixseps2":   int(ds.variables["ixseps2"][:]),
        "jyseps1_1": int(ds.variables["jyseps1_1"][:]),
        "jyseps2_1": int(ds.variables["jyseps2_1"][:]),
        "ny_inner":  int(ds.variables["ny_inner"][:]),
        "jyseps1_2": int(ds.variables["jyseps1_2"][:]),
        "jyseps2_2": int(ds.variables["jyseps2_2"][:]),
    }


def grid_format(ds, topo):
    """
    Identify the corner-geometry layout of the grid file.

    Returns one of "ingrid_dn" (rm/zm with the 2-column gap at ny_inner),
    "ingrid_sn" (rm/zm without the gap), or "hypnotoad" (Rxy_*corners).
    """
    if "rm" in ds.variables:
        n_pol = ds.variables["rm"].shape[0]
        if n_pol == topo["ny"] + 4:
            return "ingrid_dn"
        if n_pol == topo["ny"] + 2:
            return "ingrid_sn"
        raise ValueError(
            f"rm has {n_pol} poloidal columns; expected ny+4={topo['ny'] + 4} "
            f"(DN/SF) or ny+2={topo['ny'] + 2} (SN)."
        )
    if "Rxy_corners" in ds.variables:
        return "hypnotoad"
    raise KeyError("No cell-corner geometry found (need rm/zm or Rxy_corners).")


def build_cells(ds, topo, fmt):
    """
    Build the cell quadrilaterals and their BOUT (ix, iy) indices.

    Returns
    -------
    verts    : (N, 4, 2) array of cell corners (R, Z)
    ix, iy   : (N,) BOUT indices of each cell, to look up psixy[ix, iy]
    gc_verts : (M, 4, 2) array of the INGRID branch-cut guard cells at
               ny_inner (real geometry, no field value); empty otherwise
    """
    nx, ny, ny_inner = topo["nx"], topo["ny"], topo["ny_inner"]

    if fmt == "hypnotoad":
        # Corner order lower-left → lower-right → upper-right → upper-left
        names = ["", "_lower_right", "_upper_right", "_upper_left"]
        R = np.stack([ds.variables[f"Rxy{n}_corners"][:].data
                      for n in names], axis=-1)
        Z = np.stack([ds.variables[f"Zxy{n}_corners"][:].data
                      for n in names], axis=-1)
        IX, IY = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        verts = np.stack([R, Z], axis=-1).reshape(-1, 4, 2)
        return verts, IX.ravel(), IY.ravel(), np.empty((0, 4, 2))

    rm = ds.variables["rm"][:].data   # (n_pol, n_rad, 5)
    zm = ds.variables["zm"][:].data
    k = [1, 2, 4, 3]                  # NW→SW→SE→NE
    gap = 2 if fmt == "ingrid_dn" else 0

    verts, ixs, iys = [], [], []
    for ix in range(2, nx - 2):       # skip outermost 2 guard cells each side
        j_rad = ix - 1
        for iy in range(ny):
            i_pol = (iy + 1) if iy < ny_inner else (iy + 1 + gap)
            verts.append(np.column_stack([rm[i_pol, j_rad, k],
                                          zm[i_pol, j_rad, k]]))
            ixs.append(ix)
            iys.append(iy)

    gc_verts = []
    if gap:
        for ix in range(2, nx - 2):
            j_rad = ix - 1
            for i_pol_gc in (ny_inner + 1, ny_inner + 2):
                gc_verts.append(np.column_stack([rm[i_pol_gc, j_rad, k],
                                                 zm[i_pol_gc, j_rad, k]]))

    return (np.array(verts), np.array(ixs), np.array(iys),
            np.array(gc_verts).reshape(-1, 4, 2))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def find_xpoints(verts):
    """
    Locate X-points as corner nodes shared by more than 4 cells.

    Ordinary interior nodes are shared by 4 cells and boundary nodes by 1-2;
    a first-order null is shared by 8 (more for higher-order nulls). Nodes are
    merged on a lattice of _NODE_TOL_REL × (R extent).

    Returns, sorted by increasing Z: an (M, 2) array of (R, Z), the
    multiplicity of each node, and a list of the indices of the cells that
    share each node.
    """
    pts = verts.reshape(-1, 2)
    tol = _NODE_TOL_REL * np.ptp(pts[:, 0])
    keys = np.round(pts / tol).astype(np.int64)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True,
                                  return_counts=True)
    inv = inv.ravel()
    cell_of_node = inv.reshape(-1, 4)
    xpts, mult, cells = [], [], []
    for u in np.where(counts > 4)[0]:
        xpts.append(pts[inv == u].mean(axis=0))
        mult.append(counts[u])
        cells.append(np.unique(np.where(cell_of_node == u)[0]))
    if not xpts:
        return np.empty((0, 2)), np.empty(0, dtype=int), []
    xpts, mult = np.array(xpts), np.array(mult)
    order = np.argsort(xpts[:, 1])
    return xpts[order], mult[order], [cells[i] for i in order]


def build_triangulation(Rc, Zc, domain_verts):
    """
    Delaunay triangulation of the cell centres, with every triangle whose
    centroid lies outside the gridded domain (union of cell polygons) masked.
    """
    tri = mtri.Triangulation(Rc, Zc)
    cen = np.column_stack([Rc[tri.triangles].mean(axis=1),
                           Zc[tri.triangles].mean(axis=1)])
    inside = np.zeros(len(cen), dtype=bool)
    for v in domain_verts:
        # Only test centroids within this cell's bounding box (cheap prefilter)
        lo, hi = v.min(axis=0), v.max(axis=0)
        cand = np.where(~inside
                        & (cen[:, 0] >= lo[0]) & (cen[:, 0] <= hi[0])
                        & (cen[:, 1] >= lo[1]) & (cen[:, 1] <= hi[1]))[0]
        if cand.size:
            inside[cand] |= Path(v).contains_points(cen[cand])
    tri.set_mask(~inside)
    print(f"  Triangulation: {len(cen)} triangles, "
          f"{(~inside).sum()} masked (outside grid).")
    return tri


def xpoint_psi(psi_c, xpt_cells):
    """
    psi at each X-point: mean of the cell values around the X-point node.
    The separatrix through X-point i is the psi = psi_X[i] contour.
    """
    return np.array([psi_c[c].mean() for c in xpt_cells])


# ---------------------------------------------------------------------------
# Overlays (INGRID DN/SF convention, identical to visualize_in_grid.py)
# ---------------------------------------------------------------------------

def plot_branch_cuts(ax, rm, zm, topo, lw=1.5, linestyle="--"):
    """
    Overlay the topological branch cuts on the grid, drawn on the *cell edge*.

    Corner index 4 for the ``jyseps`` cuts, corner index 3 (target face) for
    ``ny_inner``. Poloidal column for a BOUT y-cut at index ``J``: ``J + 1``
    below ``ny_inner``, ``J + 3`` above it. ``jyseps1_1`` / ``jyseps2_1`` run
    out to ``ixseps1``; ``jyseps1_2`` / ``jyseps2_2`` to ``ixseps2``.
    """
    ny_inner = topo["ny_inner"]
    r = rm.transpose(1, 0, 2)
    z = zm.transpose(1, 0, 2)
    n_rad, n_pol = r.shape[0], r.shape[1]

    def ipol(J):
        return J + 1 if J < ny_inner else J + 3

    ix1, ix2 = topo.get("ixseps1"), topo.get("ixseps2")
    cut_extent = {
        "jyseps1_1": ix1, "jyseps2_1": ix1,
        "jyseps1_2": ix2, "jyseps2_2": ix2,
    }

    for name in ("jyseps1_1", "jyseps2_1", "jyseps1_2", "jyseps2_2"):
        J = topo.get(name)
        if J is None:
            continue
        ip = ipol(J)
        if not (0 <= ip < n_pol):
            continue
        ext = cut_extent[name]
        rad_max = n_rad if ext is None else min(ext, n_rad)
        ax.plot(r[0:rad_max, ip, 4], z[0:rad_max, ip, 4],
                color=_BRANCH_CUT_COLORS[name], lw=lw, ls=linestyle,
                label=f"{name}={J}", zorder=5)

    if ny_inner is not None and 0 <= ny_inner + 1 < n_pol:
        ax.plot(r[:, ny_inner + 1, 3], z[:, ny_inner + 1, 3],
                color=_BRANCH_CUT_COLORS["ny_inner"], lw=lw, ls=linestyle,
                label=f"ny_inner={ny_inner}", zorder=5)


def plot_separatrices(ax, rm, zm, topo, lw=1.5):
    """
    Overlay the separatrices (``ixseps1`` magenta, ``ixseps2`` yellow) on the
    cell edge (corner 1 of radial index ``ixseps-1``).

    ``ixseps1``: two arcs split at the ``ny_inner`` target gap.
    ``ixseps2``: core ring ``jyseps1_1+1 .. jyseps2_1`` closed on itself, lower
    leg joined ``jyseps1_1 → jyseps2_1+1``, outer half drawn continuously.
    """
    ny_inner = topo["ny_inner"]
    r = rm.transpose(1, 0, 2)
    z = zm.transpose(1, 0, 2)
    n_rad, n_pol = r.shape[0], r.shape[1]
    corner = 1

    def ipol(J):
        return J + 1 if J < ny_inner else J + 3

    def plot_cols(k, cols, color, close=False, label=None):
        cols = [c for c in cols if 0 <= c < n_pol]
        if len(cols) < 2:
            return
        R = [r[k, c, corner] for c in cols]
        Z = [z[k, c, corner] for c in cols]
        if close:
            R.append(R[0])
            Z.append(Z[0])
        ax.plot(R, Z, color=color, lw=lw, label=label, zorder=6)

    ix1 = topo.get("ixseps1")
    if ix1 is not None and 0 <= ix1 - 1 < n_rad:
        k = ix1 - 1
        plot_cols(k, range(0, ny_inner + 2), _SEP_COLORS["ixseps1"],
                  label=f"ixseps1={ix1}")
        plot_cols(k, range(ny_inner + 3, n_pol), _SEP_COLORS["ixseps1"])

    ix2 = topo.get("ixseps2")
    if ix2 is not None and 0 <= ix2 - 1 < n_rad:
        k = ix2 - 1
        j11, j21 = topo.get("jyseps1_1"), topo.get("jyseps2_1")
        plot_cols(k, range(ipol(j11 + 1), ipol(j21) + 1), _SEP_COLORS["ixseps2"],
                  close=True, label=f"ixseps2={ix2}")
        leg = list(range(0, ipol(j11) + 1)) + \
            list(range(ipol(j21 + 1), ny_inner + 2))
        plot_cols(k, leg, _SEP_COLORS["ixseps2"])
        plot_cols(k, range(ny_inner + 3, n_pol), _SEP_COLORS["ixseps2"])


def plot_separatrix_contours(ax, tri, psi_c, psi_x, lw=1.5):
    """Fallback separatrices: psi = psi_X contours, one per X-point (any topology)."""
    for i, val in enumerate(psi_x):
        color = _XPT_SEP_COLORS[i % len(_XPT_SEP_COLORS)]
        ax.tricontour(tri, psi_c, levels=[val], colors=color,
                      linewidths=lw, linestyles="solid", zorder=6)
        # Proxy artist so it shows in the legend
        ax.plot([], [], color=color, lw=lw, label=f"sep X{i} (ψ={val:.5g})")


def plot_wall(ds, ax):
    """Draw the machine wall contour if present in the file."""
    if "closed_wall_R" not in ds.variables:
        return
    Rw = ds.variables["closed_wall_R"][:].data
    Zw = ds.variables["closed_wall_Z"][:].data
    ax.plot(Rw, Zw, color="dimgray", lw=2.0, label="wall", zorder=4)


def load_limiter(path):
    """
    Load a limiter/wall contour as (R, Z) arrays: a two-column R Z text file
    (whitespace or comma, ``#``/``%`` comments) or a netCDF R/Z variable pair.
    """
    if path.endswith((".nc", ".cdf", ".netcdf")):
        lim = netCDF4.Dataset(path)
        candidates = [
            ("R", "Z"), ("r", "z"),
            ("limiter_R", "limiter_Z"),
            ("rlim", "zlim"), ("Rlim", "Zlim"),
            ("closed_wall_R", "closed_wall_Z"),
        ]
        for rname, zname in candidates:
            if rname in lim.variables and zname in lim.variables:
                R = lim.variables[rname][:].data.ravel()
                Z = lim.variables[zname][:].data.ravel()
                lim.close()
                return R, Z
        avail = list(lim.variables.keys())
        lim.close()
        raise KeyError(
            f"No recognised limiter R/Z variable pair in '{path}'. "
            f"Available variables: {avail}"
        )

    try:
        data = np.loadtxt(path, comments=("#", "%"))
    except ValueError:
        data = np.loadtxt(path, comments=("#", "%"), delimiter=",")
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(
            f"Limiter file '{path}' must have at least two columns (R, Z); "
            f"got shape {data.shape}."
        )
    return data[:, 0], data[:, 1]


def plot_limiter(R, Z, ax, lw=2.0):
    """Overlay a limiter contour."""
    ax.plot(R, Z, color="black", lw=lw, label="limiter", zorder=7)


# ---------------------------------------------------------------------------
# Zoom boxes
# ---------------------------------------------------------------------------

def bbox(points, pad):
    """(Rmin, Rmax, Zmin, Zmax) of an (N, 2) point set, padded by pad [m]."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad


def zoom_box(token, verts, Zc, xpts, args):
    """Return (Rmin, Rmax, Zmin, Zmax) for a --zoom token."""
    allpts = verts.reshape(-1, 2)

    if token == "full":
        lo, hi = allpts.min(axis=0), allpts.max(axis=0)
        mR, mZ = 0.05 * (hi[0] - lo[0]), 0.05 * (hi[1] - lo[1])
        return lo[0] - mR, hi[0] + mR, lo[1] - mZ, hi[1] + mZ

    if token == "box":
        if args.box is None:
            raise ValueError("--zoom box requires --box RMIN RMAX ZMIN ZMAX")
        return tuple(args.box)

    if token.startswith("xpt") and token != "xpoints":
        i = int(token[3:])
        if not (0 <= i < len(xpts)):
            raise ValueError(f"{token}: only {len(xpts)} X-point(s) found.")
        R0, Z0 = xpts[i]
        w = args.xpt_width
        return R0 - w, R0 + w, Z0 - w, Z0 + w

    if not len(xpts):
        raise ValueError(f"--zoom {token} needs X-points, none were found.")

    if token == "xpoints":
        box = bbox(xpts, args.pad)
        # Never narrower than the single-X-point box in either direction
        Rm, Zm = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
        hR = max(0.5 * (box[1] - box[0]), args.xpt_width)
        hZ = max(0.5 * (box[3] - box[2]), args.xpt_width)
        return Rm - hR, Rm + hR, Zm - hZ, Zm + hZ

    Zmid = 0.5 * (allpts[:, 1].min() + allpts[:, 1].max())
    if token == "legs":
        lower = xpts[xpts[:, 1] < Zmid]
        if not len(lower):
            raise ValueError("--zoom legs: no X-point below the grid mid-plane.")
        Zcut = lower[:, 1].max()
        sel = verts[Zc <= Zcut].reshape(-1, 2)
        return bbox(np.vstack([sel, lower]), args.pad)

    if token == "upper_legs":
        upper = xpts[xpts[:, 1] > Zmid]
        if not len(upper):
            raise ValueError("--zoom upper_legs: no X-point above the grid mid-plane.")
        Zcut = upper[:, 1].min()
        sel = verts[Zc >= Zcut].reshape(-1, 2)
        return bbox(np.vstack([sel, upper]), args.pad)

    raise ValueError(
        f"Unknown --zoom token '{token}'. Use full, legs, upper_legs, "
        f"xpoints, xpt<i> or box."
    )


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def plot_panel(ax, g, token, box, args, first):
    """Draw one psi panel restricted to box = (Rmin, Rmax, Zmin, Zmax)."""
    Rmin, Rmax, Zmin, Zmax = box
    psi_c = g["psi_c"]

    # Colour range: global, or from the cells whose centre is in view
    vmin, vmax = psi_c.min(), psi_c.max()
    if args.local_clim and token != "full":
        in_view = ((g["Rc"] >= Rmin) & (g["Rc"] <= Rmax)
                   & (g["Zc"] >= Zmin) & (g["Zc"] <= Zmax))
        if in_view.sum() >= 2 and np.ptp(psi_c[in_view]) > 0:
            vmin, vmax = psi_c[in_view].min(), psi_c[in_view].max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, args.levels + 2)[1:-1]

    edgecol = "k" if args.edges else "none"
    mappable = None

    if args.mode in ("heat", "both"):
        col = PatchCollection([mpatches.Polygon(v, closed=True)
                               for v in g["verts"]],
                              cmap=args.cmap, norm=norm,
                              edgecolor=edgecol, linewidth=0.15)
        col.set_array(psi_c)
        ax.add_collection(col)
        mappable = col
    elif args.edges:
        ax.add_collection(PatchCollection(
            [mpatches.Polygon(v, closed=True) for v in g["verts"]],
            facecolor="none", edgecolor="lightgray", linewidth=0.15))

    if len(g["gc_verts"]):
        ax.add_collection(PatchCollection(
            [mpatches.Polygon(v, closed=True) for v in g["gc_verts"]],
            facecolor="none",
            edgecolor=edgecol if args.mode != "contour" else "lightgray",
            linewidth=0.15))

    if args.mode == "contour":
        cs = ax.tricontour(g["tri"], psi_c, levels=levels, cmap=args.cmap,
                           norm=norm, linewidths=0.8, zorder=3)
        mappable = cs
    elif args.mode == "both":
        ax.tricontour(g["tri"], psi_c, levels=levels, colors="k",
                      linewidths=0.4, linestyles="solid", alpha=0.7, zorder=3)

    plt.colorbar(mappable, ax=ax, shrink=0.8, label=r"$\psi$ (psixy)")

    if args.plot_sep:
        if g["fmt"] == "ingrid_dn":
            plot_branch_cuts(ax, g["rm"], g["zm"], g["topo"])
            plot_separatrices(ax, g["rm"], g["zm"], g["topo"])
        else:
            plot_separatrix_contours(ax, g["tri"], psi_c, g["psi_x"])

    if args.plot_xpt:
        for i, (R0, Z0) in enumerate(g["xpts"]):
            ax.plot(R0, Z0, marker="x", color="red", ms=8, mew=2, zorder=8,
                    label="X-points" if i == 0 else None)
            if Rmin <= R0 <= Rmax and Zmin <= Z0 <= Zmax:
                ax.annotate(f"X{i}", (R0, Z0), xytext=(4, 4),
                            textcoords="offset points", color="red",
                            fontsize=9, fontweight="bold", zorder=8)

    if g["wall"] is not None:
        ax.plot(*g["wall"], color="dimgray", lw=2.0, label="wall", zorder=4)
    if g["limiter"] is not None:
        plot_limiter(*g["limiter"], ax)

    ax.set_xlim(Rmin, Rmax)
    ax.set_ylim(Zmin, Zmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"{token}")
    if first:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot psixy from a BOUT++ grid file (heat map / contours) "
                    "with leg and X-point zooms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("grid_file", help="Path to BOUT++ .grd.nc grid file")
    parser.add_argument(
        "--mode", choices=("heat", "contour", "both"), default="heat",
        help="heat map, contours, or heat map + contour lines (default: heat)",
    )
    parser.add_argument(
        "--levels", type=int, default=30,
        help="Number of contour levels per panel (default: 30)",
    )
    parser.add_argument(
        "--zoom", nargs="+", default=["full"], metavar="TOKEN",
        help="One panel per token: full, legs, upper_legs, xpoints, xpt<i>, box "
             "(default: full)",
    )
    parser.add_argument(
        "--xpt_width", type=float, default=0.06, metavar="W",
        help="Half-width [m] of the xpt<i> zoom box (default: 0.06)",
    )
    parser.add_argument(
        "--pad", type=float, default=0.02, metavar="P",
        help="Padding [m] around legs / xpoints zoom boxes (default: 0.02)",
    )
    parser.add_argument(
        "--box", type=float, nargs=4, default=None,
        metavar=("RMIN", "RMAX", "ZMIN", "ZMAX"),
        help="Manual zoom box [m] for --zoom box",
    )
    parser.add_argument(
        "--cmap", default="viridis",
        help="Matplotlib colormap (default: viridis)",
    )
    parser.add_argument(
        "--local_clim", action=argparse.BooleanOptionalAction, default=True,
        help="Rescale colours and contour levels to the cells in view on zoomed "
             "panels (use --no-local_clim for one global scale; default: True)",
    )
    parser.add_argument(
        "--edges", action=argparse.BooleanOptionalAction, default=True,
        help="Draw cell edges (use --no-edges to hide; default: True)",
    )
    parser.add_argument(
        "--plot_sep", action=argparse.BooleanOptionalAction, default=True,
        help="Overlay separatrices and branch cuts "
             "(use --no-plot_sep to disable; default: True)",
    )
    parser.add_argument(
        "--plot_xpt", action=argparse.BooleanOptionalAction, default=True,
        help="Mark the X-points found from the grid geometry "
             "(use --no-plot_xpt to disable; default: True)",
    )
    parser.add_argument(
        "--limiter", default=None, metavar="PATH",
        help="Overlay a limiter contour read from PATH (two-column R Z text "
             "file, or a netCDF file with an R/Z variable pair).",
    )
    parser.add_argument(
        "--save", default=None, metavar="FILE",
        help="Save figure to FILE instead of displaying interactively",
    )
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    ds   = load_grid(args.grid_file)
    topo = get_topology_indices(ds)
    fmt  = grid_format(ds, topo)

    print("Grid file:", args.grid_file, f"  [{fmt}]")
    print(f"  nx={topo['nx']}, ny={topo['ny']}")
    print(f"  ixseps1={topo['ixseps1']}, ixseps2={topo['ixseps2']}")
    print(f"  jyseps1_1={topo['jyseps1_1']}, jyseps2_1={topo['jyseps2_1']}, "
          f"ny_inner={topo['ny_inner']}, jyseps1_2={topo['jyseps1_2']}, "
          f"jyseps2_2={topo['jyseps2_2']}")

    psi = ds.variables["psixy"][:].data      # (nx, ny)
    verts, ix, iy, gc_verts = build_cells(ds, topo, fmt)
    Rc, Zc = verts[..., 0].mean(axis=1), verts[..., 1].mean(axis=1)
    psi_c = psi[ix, iy]
    print(f"  psixy range: [{psi_c.min():.6g}, {psi_c.max():.6g}]")

    xpts, mult, xpt_cells = find_xpoints(verts)
    psi_x = xpoint_psi(psi_c, xpt_cells)
    print(f"  X-points found (sorted by Z): {len(xpts)}")
    for i, ((R0, Z0), m, px) in enumerate(zip(xpts, mult, psi_x)):
        print(f"    X{i}: R={R0:.4f} m, Z={Z0:.4f} m, psi_X={px:.6g}  "
              f"({m} cells share the node)")

    tri = None
    if args.mode in ("contour", "both") or (args.plot_sep and fmt != "ingrid_dn"):
        print("Building triangulation...")
        domain = np.concatenate([verts, gc_verts]) if len(gc_verts) else verts
        tri = build_triangulation(Rc, Zc, domain)

    g = {
        "topo": topo, "fmt": fmt, "verts": verts, "gc_verts": gc_verts,
        "Rc": Rc, "Zc": Zc, "psi_c": psi_c, "xpts": xpts, "tri": tri,
        "psi_x": psi_x,
        "rm": ds.variables["rm"][:].data if "rm" in ds.variables else None,
        "zm": ds.variables["zm"][:].data if "zm" in ds.variables else None,
        "wall": ((ds.variables["closed_wall_R"][:].data,
                  ds.variables["closed_wall_Z"][:].data)
                 if "closed_wall_R" in ds.variables else None),
        "limiter": load_limiter(args.limiter) if args.limiter else None,
    }

    boxes = [zoom_box(t, verts, Zc, xpts, args) for t in args.zoom]

    n = len(args.zoom)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n if n > 1 else 7, 9),
                             squeeze=False)
    for i, (token, box) in enumerate(zip(args.zoom, boxes)):
        print(f"Panel '{token}': R=[{box[0]:.3f}, {box[1]:.3f}], "
              f"Z=[{box[2]:.3f}, {box[3]:.3f}]")
        plot_panel(axes[0, i], g, token, box, args, first=(i == 0))

    fig.suptitle(f"psixy — {os.path.basename(args.grid_file)}")
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()

    ds.close()


if __name__ == "__main__":
    main()
