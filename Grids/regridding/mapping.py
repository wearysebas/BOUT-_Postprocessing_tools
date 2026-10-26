"""Mapping of fields between grids along field lines"""

import numpy as np

from .coordinates import describe, match_targets
from .physics import (
    apply_floors,
    cell_volume,
    conserve,
    default_floors,
    from_primitive,
    inventories,
    species_fields,
    to_primitive,
)


def _along(piece, key, values, c, periodic):
    """
    Values of one old piece at along-tube coordinate c, clamped at the ends
    """
    coord = piece[key]
    v = values[piece["x"], piece["y"]]
    flat = v.reshape(len(coord), -1)
    if periodic:
        order = np.argsort(coord)
        coord = coord[order]
        flat = flat[order]
        coord = np.concatenate([coord[-1:] - 2 * np.pi, coord, coord[:1] + 2 * np.pi])
        flat = np.concatenate([flat[-1:], flat, flat[:1]])
        c = (c - coord[1]) % (2 * np.pi) + coord[1]
    elif coord[0] > coord[-1]:
        coord = coord[::-1]
        flat = flat[::-1]
    out = np.stack([np.interp(c, coord, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((len(c),) + v.shape[1:])


def tube_interpolate(sources, key, values, rho, c, periodic=False):
    """
    Interpolate values at (rho, c) from a family of old pieces: 1D along the
    two pieces that bracket rho, then linear in rho. Clamped outside the
    rho range of the family

    Returns
    -------
    array [len(c), ...], and the distance of rho outside the family's range
    """
    rhos = np.array([p["rho"] for p in sources])
    order = np.argsort(rhos)
    rhos = rhos[order]
    sources = [sources[i] for i in order]
    if rho <= rhos[0] or len(sources) == 1:
        return _along(sources[0], key, values, c, periodic), max(rhos[0] - rho, 0.0)
    if rho >= rhos[-1]:
        return _along(sources[-1], key, values, c, periodic), rho - rhos[-1]
    i = np.searchsorted(rhos, rho)
    w = (rho - rhos[i - 1]) / (rhos[i] - rhos[i - 1])
    a = _along(sources[i - 1], key, values, c, periodic)
    b = _along(sources[i], key, values, c, periodic)
    return (1 - w) * a + w * b, 0.0


def map_pieces(old, new, prim, match):
    """
    Map every primitive field onto the new grid, three ways

    Returns
    -------
    f_up : closed pieces in (rho, theta); open pieces in (rho, lam), lam the
        poloidal distance from the X-point cut
    f_div : open pieces in (rho, ell_T), ell_T the normalised parallel
        distance from the target; closed pieces as f_up
    lam : lam of every new cell, +inf on closed field lines [nx, ny]
    """
    shape = new["shape"]
    f_up = {name: np.full(shape + v.shape[2:], np.nan) for name, v in prim.items()}
    f_div = {name: np.full(shape + v.shape[2:], np.nan) for name, v in prim.items()}
    lam = np.full(shape, np.inf)

    old_closed = [p for p in old["pieces"] if p["target"] is None and p["physical"]]
    old_by_face = {}
    for p in old["pieces"]:
        if p["target"] is not None and p["physical"]:
            old_by_face.setdefault(p["target"], []).append(p)
    old_open = [p for ps in old_by_face.values() for p in ps]

    for p in new["pieces"]:
        x, ys = p["x"], p["y"]
        if p["target"] is None:
            sources = old_closed if old_closed else old_open
            key, c, periodic = ("theta", p["theta"], True) if old_closed else ("lam", np.zeros(len(ys)), False)
            for name, v in prim.items():
                val, _ = tube_interpolate(sources, key, v, p["rho"], c, periodic)
                f_up[name][x, ys] = val
                f_div[name][x, ys] = val
            continue
        sources = old_by_face.get(match[p["target"]], old_open)
        lam[x, ys] = p["lam"]
        for name, v in prim.items():
            val_up, _ = tube_interpolate(sources, "lam", v, p["rho"], p["lam"])
            val_div, _ = tube_interpolate(sources, "ell_T", v, p["rho"], p["ell_T"])
            f_up[name][x, ys] = val_up
            f_div[name][x, ys] = val_div
    return f_up, f_div, lam


def tanh_step(u):
    """
    1/2 [1 + tanh(u)]
    """
    return 0.5 * (1 + np.tanh(u))


def blend(a, b, w):
    """
    w a + (1 - w) b for every field, w [nx, ny]
    """
    out = {}
    for name in a:
        ww = w.reshape(w.shape + (1,) * (a[name].ndim - 2))
        out[name] = ww * a[name] + (1 - ww) * b[name]
    return out


def divertor_blend(lambda0=0.0, width=None):
    """
    Upstream map f_up and leg map f_div blended across the X-point:
    f = w f_up + (1 - w) f_div, w = 1/2 [1 + tanh((lam - lambda0) / width)],
    w = 1 on closed field lines
    """

    def f(old, new, prim, mxg):
        f_up, f_div, lam = map_pieces(old, new, prim, match_targets(old, new))
        wd = new["width_pol"] if width is None else width
        w = np.where(np.isinf(lam), 1.0, tanh_step((lam - lambda0) / wd))
        return blend(f_up, f_div, w)

    return f


def regrid_fields(
    old_grid,
    new_grid,
    fields,
    mxg=2,
    lambda0=0.0,
    width_pol=None,
    floors=None,
    conserve_scope="global",
    conserve_mode="adaptive",
    volume_threshold=0.05,
):
    """
    Map restart fields [nx, ny, nz] from old_grid onto new_grid

    lambda0, width_pol : centre and width [m] of the tanh blending the
        upstream and leg maps, in poloidal distance from the X-point.
        Default width: 3 poloidal cells at the X-point of the new grid

    floors : dict field -> floor. Default half the smallest positive old value

    conserve_scope : "global", "closed/open" or None (no rescaling)
    conserve_mode : "adaptive", "totals" or "averages". "totals" keeps the
        particle and energy content, "averages" keeps it per unit volume,
        "adaptive" uses totals if the volume changes by less than
        volume_threshold (relative), averages otherwise

    Returns
    -------
    new_fields, report
    """
    names = list(fields)
    species, other = species_fields(names)
    if floors is None:
        floors = {name: 0.5 * value for name, value in default_floors(fields, species).items()}
    if conserve_mode not in ("adaptive", "totals", "averages"):
        raise ValueError(f"Unknown conserve_mode '{conserve_mode}'")

    def conserve_target(inventory_old, volume_old, volume_new):
        ratio = volume_new / volume_old
        averages = conserve_mode == "averages" or (
            conserve_mode == "adaptive" and abs(ratio - 1) >= volume_threshold
        )
        scale = ratio if averages else 1.0
        target = {k: v * scale for k, v in inventory_old.items() if k != "volume"}
        return target, "averages" if averages else "totals"

    old = describe(old_grid, mxg)
    new = describe(new_grid, mxg)
    prim = to_primitive(fields, species, other, floors)
    new_prim = divertor_blend(lambda0, width_pol)(old, new, prim, mxg)
    new_fields = from_primitive(new_prim, species, other)

    vol_old = cell_volume(old_grid, mxg)
    vol_new = cell_volume(new_grid, mxg)
    report = {"inventory_old": inventories(fields, species, vol_old)}
    report["inventory_mapped"] = inventories(new_fields, species, vol_new)

    old_closed = old["coords"]["closed"]
    new_closed = new["coords"]["closed"]
    if conserve_scope == "global":
        target, used = conserve_target(
            report["inventory_old"],
            report["inventory_old"]["volume"],
            report["inventory_mapped"]["volume"],
        )
        report["conserve_mode"] = used
        report["factors"] = conserve(new_fields, species, target, vol_new)
    elif conserve_scope == "closed/open":
        report["factors"] = {}
        report["conserve_mode"] = {}
        for label, m_old, m_new in (
            ("closed", old_closed, new_closed),
            ("open", ~old_closed, ~new_closed),
        ):
            if m_old.any() and m_new.any():
                inv_old = inventories(fields, species, vol_old, m_old)
                inv_new = inventories(new_fields, species, vol_new, m_new)
                target, used = conserve_target(inv_old, inv_old["volume"], inv_new["volume"])
                report["conserve_mode"][label] = used
                report["factors"][label] = conserve(new_fields, species, target, vol_new, m_new)
    elif conserve_scope is not None:
        raise ValueError(f"Unknown conserve_scope '{conserve_scope}'")

    report["clipped"] = apply_floors(new_fields, floors)
    report["inventory_new"] = inventories(new_fields, species, vol_new)
    return new_fields, report
