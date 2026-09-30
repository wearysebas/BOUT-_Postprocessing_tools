"""Topology identification and field line connectivity, following BoutMesh"""

import warnings

import numpy as np


TOPOLOGY_FAMILIES = ("CFL", "SN", "UDN", "CDN", "SF", "XPT")


SF_TYPES = ("SF_plus_LFS", "SF_plus_HFS", "SF_minus_LFS", "SF_minus_HFS", "XPT", "SF")


def read_topology(grid):
    """
    Identify the topology of a grid in the same way as BoutMesh
    (readIngridTopology, getMeshTopology, getSnowflakeType)

    Unlike BoutMesh, a missing or unrecognised 'topology' tag raises
    instead of falling back on the separatrix indices

    Returns
    -------
    (family, sf_type) : family in TOPOLOGY_FAMILIES, sf_type in SF_TYPES
        for SF/XPT grids, None otherwise
    """
    try:
        tag = str(grid["topology"]).strip().upper()
    except KeyError:
        tag = ""
    if not tag:
        raise ValueError("Grid has no 'topology' variable")

    if tag in ("CLOSED_FIELD_LINE", "CFL"):
        family = "CFL"
    elif "SINGLE_NULL" in tag or tag == "SN":
        family = "SN"
    elif tag in ("UNCONNECTED_DOUBLE_NULL", "UDN"):
        family = "UDN"
    elif tag in ("CONNECTED_DOUBLE_NULL", "CDN"):
        family = "CDN"
    elif "SF" in tag or "SNOWFLAKE" in tag:
        family = "SF"
    elif "XPOINT_TARGET" in tag or "XPT" in tag:
        family = "XPT"
    else:
        raise ValueError(f"Unrecognised 'topology' value '{tag}' in grid")

    if family != "SF":
        return family, None

    if "105" in tag or "135" in tag:
        sf_type = "SF_plus_HFS"
    elif "165" in tag:
        sf_type = "SF_minus_HFS"
    elif "15" in tag:
        sf_type = "SF_minus_LFS"
    elif "45" in tag or "75" in tag:
        sf_type = "SF_plus_LFS"
    elif "TARGET" in tag:
        sf_type = "XPT"
    else:
        warnings.warn(
            f"Snowflake type not specified by topology '{tag}'; "
            "treated as snowflake+ LFS, as BoutMesh does"
        )
        sf_type = "SF"
    return family, sf_type


def y_decomposition_indices(grid):
    """
    Separatrix indices after the checks in BoutMesh::setYDecompositionIndices
    """
    ny = int(grid["ny"])
    jyseps1_1 = int(grid["jyseps1_1"])
    jyseps2_1 = int(grid["jyseps2_1"])
    jyseps1_2 = int(grid["jyseps1_2"])
    jyseps2_2 = int(grid["jyseps2_2"])
    ny_inner = int(grid["ny_inner"])

    jyseps1_1 = max(jyseps1_1, -1)
    if jyseps2_1 < jyseps1_1:
        jyseps2_1 = jyseps1_1 + 1
    if jyseps1_2 < jyseps2_1:
        jyseps1_2 = jyseps2_1
    if jyseps2_2 >= ny:
        jyseps2_2 = ny - 1
    if jyseps2_2 < jyseps1_2:
        if jyseps1_2 >= ny:
            raise ValueError(f"jyseps1_2 ({jyseps1_2}) must be < ny ({ny})")
        jyseps2_2 = jyseps1_2

    if jyseps1_1 < 0 and jyseps2_2 >= ny - 1:
        numberOfXPoints = 0
    elif jyseps2_1 == jyseps1_2:
        numberOfXPoints = 1
    else:
        numberOfXPoints = 2

    return {
        "jyseps1_1": jyseps1_1,
        "jyseps2_1": jyseps2_1,
        "jyseps1_2": jyseps1_2,
        "jyseps2_2": jyseps2_2,
        "ny_inner": ny_inner,
        "numberOfXPoints": numberOfXPoints,
    }


def ixseps_lower_upper(grid, family):
    """
    ixseps_lower and ixseps_upper as set in BoutMesh::topology()
    """
    ixseps1 = int(grid["ixseps1"])
    ixseps2 = int(grid["ixseps2"])
    if family in ("SN", "CFL"):
        return ixseps1, ixseps1
    if family in ("UDN", "CDN"):
        return ixseps1, ixseps2
    if ixseps1 == ixseps2:
        raise ValueError("Snowflake topology can't have the two same separatrices")
    return min(ixseps1, ixseps2), max(ixseps1, ixseps2)


def y_links(grid):
    """
    Port of the connections made in BoutMesh::topology()

    Returns
    -------
    links : list of (top, bottom, xlt) in the order BoutMesh applies them.
        Cells x < xlt at y = top connect upwards to y = bottom
    targets : list of (y, xge, xlt). No connection between y and y + 1
    """
    family, sf_type = read_topology(grid)
    ind = y_decomposition_indices(grid)
    jyseps1_1 = ind["jyseps1_1"]
    jyseps2_1 = ind["jyseps2_1"]
    jyseps1_2 = ind["jyseps1_2"]
    jyseps2_2 = ind["jyseps2_2"]
    ny_inner = ind["ny_inner"]
    nx = int(grid["nx"])

    if family == "XPT":
        raise ValueError("BoutMesh::topology() does not support XPoint_target")
    ixseps_lower, ixseps_upper = ixseps_lower_upper(grid, family)

    if family in ("SN", "CFL"):
        links = [
            (jyseps2_2, jyseps1_1 + 1, ixseps_lower),
            (jyseps1_1, jyseps2_2 + 1, ixseps_lower),
        ]
        return links, []

    if family in ("UDN", "CDN"):
        links = [
            (jyseps2_2, jyseps1_1 + 1, ixseps_lower),
            (jyseps1_1, jyseps2_2 + 1, ixseps_lower),
            (jyseps2_1, jyseps1_2 + 1, ixseps_upper),
            (jyseps1_2, jyseps2_1 + 1, ixseps_upper),
        ]
    elif sf_type in ("SF_plus_LFS", "SF"):
        links = [
            (jyseps2_2, jyseps1_2 + 1, ixseps_lower),
            (jyseps1_2, jyseps2_2 + 1, ixseps_lower),
            (jyseps2_1, jyseps1_1 + 1, ixseps_upper),
            (jyseps1_1, jyseps2_1 + 1, ixseps_upper),
        ]
    elif sf_type in ("SF_plus_HFS", "SF_minus_LFS"):
        links = [
            (jyseps2_2, jyseps1_1 + 1, ixseps_lower),
            (jyseps1_1, jyseps2_2 + 1, ixseps_lower),
            (jyseps2_1, jyseps1_2 + 1, ixseps_upper),
            (jyseps1_2, jyseps2_1 + 1, ixseps_upper),
        ]
    elif sf_type == "SF_minus_HFS":
        links = [
            (jyseps2_2, jyseps1_1 + 1, ixseps_upper),
            (jyseps1_1, jyseps2_2 + 1, ixseps_upper),
            (jyseps2_1, jyseps1_2 + 1, ixseps_lower),
            (jyseps1_2, jyseps2_1 + 1, ixseps_lower),
        ]
    else:
        raise ValueError(f"BoutMesh::topology() makes no connections for {sf_type}")
    return links, [(ny_inner - 1, 0, nx)]


def field_line_connectivity(grid):
    """
    Global y-neighbours of every cell, following BoutMesh::topology()

    Returns
    -------
    up, down : int arrays [nx, ny]. y index of the next cell along the
        field line in +y / -y, or -1 at a target
    """
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    links, targets = y_links(grid)

    up = np.tile(np.arange(1, ny + 1), (nx, 1))
    up[:, -1] = -1
    down = np.tile(np.arange(-1, ny - 1), (nx, 1))

    for top, bottom, xlt in links:
        if xlt <= 0 or not (0 <= top < ny and 0 <= bottom < ny):
            continue
        up[:xlt, top] = bottom
        up[xlt:, top] = top + 1 if top + 1 < ny else -1
        down[:xlt, bottom] = top
        down[xlt:, bottom] = bottom - 1

    for y, xge, xlt in targets:
        up[xge:xlt, y] = -1
        down[xge:xlt, y + 1] = -1

    for x in range(nx):
        for y in range(ny):
            if up[x, y] >= 0 and down[x, up[x, y]] != y:
                raise ValueError(f"Inconsistent y connection at x={x}, y={y}")
            if down[x, y] >= 0 and up[x, down[x, y]] != y:
                raise ValueError(f"Inconsistent y connection at x={x}, y={y}")
    return up, down


def flux_tubes(grid):
    """
    Field lines of the grid, one per x index and connected set of y

    Returns
    -------
    list of dict with keys
    - "x" : x index
    - "y" : list of y indices ordered along +y
    - "closed" : True for closed field lines
    - "lower", "upper" : y index of the first / last cell of an open
      field line (the cells next to the targets), None if closed
    """
    up, down = field_line_connectivity(grid)
    nx, ny = up.shape
    tubes = []
    for x in range(nx):
        seen = np.zeros(ny, dtype=bool)
        for y0 in range(ny):
            if seen[y0]:
                continue
            y = y0
            for _ in range(ny):
                if down[x, y] < 0 or down[x, y] == y0:
                    break
                y = down[x, y]
            closed = down[x, y] == y0 and down[x, y] >= 0
            start = y0 if closed else y
            ys = [start]
            while up[x, ys[-1]] >= 0 and up[x, ys[-1]] != start:
                ys.append(up[x, ys[-1]])
            seen[ys] = True
            tubes.append(
                {
                    "x": x,
                    "y": ys,
                    "closed": closed,
                    "lower": None if closed else ys[0],
                    "upper": None if closed else ys[-1],
                }
            )
    return tubes


def topology_regions(grid):
    """
    Split the grid into logically rectangular regions with uniform
    connectivity. Same description as regions(), plus
    - "closed" : True if on closed field lines
    - "lower_target", "upper_target" : y index of the target face cell
      at each end of the field lines, None if not open
    """
    up, down = field_line_connectivity(grid)
    nx, ny = up.shape

    ycuts = [y for y in range(ny - 1) if (up[:, y] != y + 1).any()]
    xcuts = [
        x
        for x in range(1, nx)
        if (up[x - 1] != up[x]).any() or (down[x - 1] != down[x]).any()
    ]
    ybounds = [0] + [y + 1 for y in ycuts] + [ny]
    xbounds = [0] + xcuts + [nx]

    closed = np.zeros((nx, ny), dtype=bool)
    lower_end = np.full((nx, ny), -1)
    upper_end = np.full((nx, ny), -1)
    for tube in flux_tubes(grid):
        closed[tube["x"], tube["y"]] = tube["closed"]
        if not tube["closed"]:
            lower_end[tube["x"], tube["y"]] = tube["lower"]
            upper_end[tube["x"], tube["y"]] = tube["upper"]

    def block_of(x, y):
        i = np.searchsorted(xbounds, x, side="right") - 1
        j = np.searchsorted(ybounds, y, side="right") - 1
        return f"x{i}y{j}"

    regions = {}
    for i in range(len(xbounds) - 1):
        for j in range(len(ybounds) - 1):
            xf, xl = xbounds[i], xbounds[i + 1] - 1
            yf, yl = ybounds[j], ybounds[j + 1] - 1
            xs = range(xf, xl + 1)

            def neighbour(nbr_y):
                names = {None if nbr_y[x] < 0 else block_of(x, nbr_y[x]) for x in xs}
                if len(names) != 1:
                    raise ValueError(f"Region x{i}y{j} connects to several regions")
                return names.pop()

            if len({bool(v) for v in closed[xf : xl + 1, yf : yl + 1].flat}) != 1:
                raise ValueError(f"Region x{i}y{j} mixes open and closed field lines")
            is_closed = bool(closed[xf, yf])
            regions[f"x{i}y{j}"] = {
                "inner": block_of(xf - 1, yf) if xf > 0 else None,
                "outer": block_of(xl + 1, yf) if xl < nx - 1 else None,
                "lower": neighbour(down[:, yf]),
                "upper": neighbour(up[:, yl]),
                "xfirst": xf,
                "xlast": xl,
                "yfirst": yf,
                "ylast": yl,
                "closed": is_closed,
                "lower_target": None if is_closed else int(lower_end[xf, yf]),
                "upper_target": None if is_closed else int(upper_end[xf, yf]),
            }
    return regions
