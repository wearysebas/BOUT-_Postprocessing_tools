"""Topology-independent coordinates of the cells of a grid"""

import numpy as np

from .topology import field_line_connectivity, flux_tubes


GEOMETRY_VARS = ("Rxy", "Zxy", "psixy", "Bxy", "Bpxy", "hthe", "dx", "dy", "J")


def grid_coordinates(grid):
    """
    Topology-independent coordinates of every cell of a grid

    Returns
    -------
    dict of arrays [nx, ny]
    - "rho" : midplane-equivalent distance from the separatrix [m],
      (psi - psi_sep) / (R Bp) at the outboard midplane separatrix
    - "closed" : True on closed field lines
    - "s_par", "l_pol" : parallel / poloidal distance from the lower
      target of open field lines [m]
    - "L_par", "L_pol" : parallel / poloidal length of the field line [m]
    - "ell" : s_par / L_par, 0 at the lower target, 1 at the upper
    - "theta" : poloidal angle about the magnetic axis
    - "lower_target", "upper_target" : y index of the end cells of open
      field lines, -1 on closed field lines
    and scalars "psi_sep", "R_omp", "Z_omp", "R_axis", "Z_axis"
    """
    geom = {name: np.asarray(grid[name], dtype=float) for name in GEOMETRY_VARS}
    nx, ny = geom["Rxy"].shape
    dl_pol = geom["hthe"] * geom["dy"]
    ds_par = dl_pol * geom["Bxy"] / geom["Bpxy"]

    tubes = flux_tubes(grid)

    closed = np.zeros((nx, ny), dtype=bool)
    s_par = np.full((nx, ny), np.nan)
    l_pol = np.full((nx, ny), np.nan)
    L_par = np.full((nx, ny), np.nan)
    L_pol = np.full((nx, ny), np.nan)
    lower_target = np.full((nx, ny), -1)
    upper_target = np.full((nx, ny), -1)

    for tube in tubes:
        x = tube["x"]
        ys = np.array(tube["y"])
        dl = dl_pol[x, ys]
        ds = ds_par[x, ys]
        l_centre = np.cumsum(dl) - 0.5 * dl
        s_centre = np.cumsum(ds) - 0.5 * ds
        closed[x, ys] = tube["closed"]
        L_pol[x, ys] = dl.sum()
        L_par[x, ys] = ds.sum()
        if not tube["closed"]:
            s_par[x, ys] = s_centre
            l_pol[x, ys] = l_centre
            lower_target[x, ys] = ys[0]
            upper_target[x, ys] = ys[-1]

    psi = geom["psixy"]
    R = geom["Rxy"]
    Z = geom["Zxy"]
    closed_x = np.where(closed.any(axis=1))[0]
    if len(closed_x) == 0:
        raise ValueError("Grid has no closed field lines to define the separatrix")
    x_in = closed_x.min()
    x_sep = closed_x.max()
    core_y = np.where(closed[x_sep])[0]
    y_omp = core_y[np.argmax(R[x_sep, core_y])]

    R_axis = R[x_in, closed[x_in]].mean()
    Z_axis = Z[x_in, closed[x_in]].mean()

    if x_sep + 1 < nx:
        psi_sep = 0.5 * (psi[x_sep, y_omp] + psi[x_sep + 1, y_omp])
        grad_psi = 0.5 * (
            R[x_sep, y_omp] * geom["Bpxy"][x_sep, y_omp]
            + R[x_sep + 1, y_omp] * geom["Bpxy"][x_sep + 1, y_omp]
        )
        R_omp = 0.5 * (R[x_sep, y_omp] + R[x_sep + 1, y_omp])
        Z_omp = 0.5 * (Z[x_sep, y_omp] + Z[x_sep + 1, y_omp])
    else:
        psi_sep = psi[x_sep, y_omp]
        grad_psi = R[x_sep, y_omp] * geom["Bpxy"][x_sep, y_omp]
        R_omp = R[x_sep, y_omp]
        Z_omp = Z[x_sep, y_omp]

    direction = np.sign(psi[-1, y_omp] - psi[0, y_omp])
    rho = direction * (psi - psi_sep) / grad_psi

    return {
        "rho": rho,
        "closed": closed,
        "s_par": s_par,
        "l_pol": l_pol,
        "L_par": L_par,
        "L_pol": L_pol,
        "ell": s_par / L_par,
        "theta": np.arctan2(Z - Z_axis, R - R_axis),
        "lower_target": lower_target,
        "upper_target": upper_target,
        "psi_sep": psi_sep,
        "R_omp": R_omp,
        "Z_omp": Z_omp,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
    }


def half_tubes(grid, coords, mxg):
    """
    Split field lines into pieces that share one set of along-tube
    coordinates

    Closed field lines are one piece with target None. Open field lines are
    split at their midpoint into a lower and an upper half, each belonging to
    its nearest target face ("lower", y) or ("upper", y)

    Returns
    -------
    list of dict with keys "x", "y" (y indices), "target", "rho" (scalar),
    "theta", "ell_T", "lam" (arrays along the piece), "physical" (bool)
    """
    R = np.asarray(grid["Rxy"], dtype=float)
    nx, ny = R.shape
    dl_pol = np.asarray(grid["hthe"], dtype=float) * np.asarray(grid["dy"], dtype=float)
    up, _ = field_line_connectivity(grid)
    cut_rows = {y for y in range(ny - 1) if (up[:, y] != y + 1).any()}

    pieces = []
    for tube in flux_tubes(grid):
        x = tube["x"]
        ys = np.array(tube["y"])
        rho = float(np.mean(coords["rho"][x, ys]))
        physical = mxg <= x < nx - mxg
        if tube["closed"]:
            pieces.append(
                {
                    "x": x,
                    "y": ys,
                    "target": None,
                    "rho": rho,
                    "theta": coords["theta"][x, ys],
                    "physical": physical,
                }
            )
            continue

        dl = dl_pol[x, ys]
        l_centre = np.cumsum(dl) - 0.5 * dl
        L = dl.sum()
        crossings = np.cumsum(dl)[:-1][[y in cut_rows for y in ys[:-1]]]
        s = coords["s_par"][x, ys]
        L_par = coords["L_par"][x, ys[0]]
        near_lower = l_centre <= 0.5 * L

        for lower in (True, False):
            k = near_lower if lower else ~near_lower
            if not k.any():
                continue
            if lower:
                d = l_centre[k]
                leg = crossings.min() if len(crossings) else 0.0
                ell = s[k] / L_par
                face = ("lower", int(ys[0]))
                order = slice(None)
            else:
                d = L - l_centre[k]
                leg = L - crossings.max() if len(crossings) else 0.0
                ell = (L_par - s[k]) / L_par
                face = ("upper", int(ys[-1]))
                order = slice(None, None, -1)
            pieces.append(
                {
                    "x": x,
                    "y": ys[k][order],
                    "target": face,
                    "rho": rho,
                    "ell_T": ell[order],
                    "lam": (d - leg)[order],
                    "d_target": d[order],
                    "physical": physical,
                }
            )
    return pieces


def target_faces(grid, pieces):
    """
    (R, Z) centroid of the cells next to every target face
    """
    R = np.asarray(grid["Rxy"], dtype=float)
    Z = np.asarray(grid["Zxy"], dtype=float)
    cells = {}
    for p in pieces:
        if p["target"] is not None:
            cells.setdefault(p["target"], []).append((p["x"], p["y"][0]))
    return {
        face: (float(np.mean([R[c] for c in cs])), float(np.mean([Z[c] for c in cs])))
        for face, cs in cells.items()
    }


def poloidal_width(grid):
    """
    Default tanh width: 3 x median poloidal cell length on the X-point cut rows
    """
    dl_pol = np.asarray(grid["hthe"], dtype=float) * np.asarray(grid["dy"], dtype=float)
    ny = dl_pol.shape[1]
    up, _ = field_line_connectivity(grid)
    rows = sorted({y for y in range(ny - 1) if (up[:, y] != y + 1).any()})
    if not rows:
        return np.inf
    cells = np.concatenate([dl_pol[:, y] for y in rows] + [dl_pol[:, y + 1] for y in rows])
    return 3.0 * float(np.median(cells))


def describe(grid, mxg):
    """
    Coordinates, field line pieces, target faces and default tanh width of a grid
    """
    coords = grid_coordinates(grid)
    pieces = half_tubes(grid, coords, mxg)
    return {
        "coords": coords,
        "pieces": pieces,
        "faces": target_faces(grid, pieces),
        "width_pol": poloidal_width(grid),
        "shape": coords["rho"].shape,
    }


def match_targets(old, new):
    """
    Each new target face -> nearest old target face in (R, Z)
    """
    old_faces = list(old["faces"])
    old_RZ = np.array([old["faces"][f] for f in old_faces])
    match = {}
    for face, RZ in new["faces"].items():
        dist = np.hypot(*(old_RZ - np.array(RZ)).T)
        match[face] = old_faces[int(np.argmin(dist))]
    return match
