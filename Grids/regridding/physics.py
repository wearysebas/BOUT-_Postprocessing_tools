"""Primitive variables, floors and conservation of the evolved fields"""

import numpy as np


def species_fields(names):
    """
    Group Hermes-3 evolved fields by species

    Returns
    -------
    species : dict name -> {"N": field or None, "P": field or None,
        "NV": field or None, "Z": charge}. Electrons ("e") have no N;
        their density comes from quasineutrality
    other : list of fields not belonging to a species
    """
    names = set(names)
    species = {}
    for name in names:
        for prefix in ("NV", "N", "P"):
            if name.startswith(prefix) and len(name) > len(prefix):
                s = name[len(prefix):]
                if prefix == "N" and s.startswith("V"):
                    continue
                species.setdefault(s, {"N": None, "P": None, "NV": None})
                species[s][prefix] = name
                break
    species = {
        s: f for s, f in species.items() if f["N"] is not None or s == "e"
    }
    for s, f in species.items():
        f["Z"] = -1 if s == "e" else s.count("+") - s.count("-")
    used = {v for f in species.values() for k, v in f.items() if k != "Z" and v}
    return species, sorted(names - used)


def has_ions(species):
    return any(s != "e" and f["Z"] > 0 for s, f in species.items())


def electron_density(fields, species):
    """
    Quasineutral electron density, sum of Z N over ion species
    """
    ne = 0.0
    for s, f in species.items():
        if s != "e" and f["Z"] > 0:
            ne = ne + f["Z"] * fields[f["N"]]
    return ne


def to_primitive(fields, species, other, floors):
    """
    Variables that are interpolated between grids:
    log N, log T = log(P / N), V = NV / N, and the other fields as they are.
    Electrons without ion species (fixed density) map log P and NV
    """
    prim = {}
    ions = has_ions(species)
    for s, f in species.items():
        if s == "e" and not ions:
            if f["P"]:
                prim["logP_e"] = np.log(np.maximum(fields[f["P"]], floors[f["P"]]))
            if f["NV"]:
                prim["NV_e"] = fields[f["NV"]]
            continue
        if s == "e":
            n = electron_density(fields, species)
        else:
            n = np.maximum(fields[f["N"]], floors[f["N"]])
            prim["logN_" + s] = np.log(n)
        if f["P"]:
            p = np.maximum(fields[f["P"]], floors[f["P"]])
            prim["logT_" + s] = np.log(p / n)
        if f["NV"]:
            prim["V_" + s] = fields[f["NV"]] / n
    for name in other:
        prim[name] = fields[name]
    return prim


def from_primitive(prim, species, other):
    """
    Rebuild N, P, NV from the interpolated primitive variables
    """
    fields = {}
    density = {}
    for s, f in species.items():
        if s != "e":
            density[s] = np.exp(prim["logN_" + s])
            fields[f["N"]] = density[s]
    ne = sum(f["Z"] * density[s] for s, f in species.items() if s != "e" and f["Z"] > 0)
    ions = has_ions(species)
    for s, f in species.items():
        if s == "e" and not ions:
            if f["P"]:
                fields[f["P"]] = np.exp(prim["logP_e"])
            if f["NV"]:
                fields[f["NV"]] = prim["NV_e"]
            continue
        n = ne if s == "e" else density[s]
        if f["P"]:
            fields[f["P"]] = n * np.exp(prim["logT_" + s])
        if f["NV"]:
            fields[f["NV"]] = n * prim["V_" + s]
    for name in other:
        fields[name] = prim[name]
    return fields


def default_floors(fields, species):
    """
    Smallest positive value of every density and pressure field
    """
    floors = {}
    for f in species.values():
        for key in ("N", "P"):
            name = f[key]
            if name:
                positive = fields[name][fields[name] > 0]
                floors[name] = positive.min() if positive.size else 0.0
    return floors


def apply_floors(fields, floors):
    clipped = {}
    for name, floor in floors.items():
        if name in fields:
            low = fields[name] < floor
            clipped[name] = int(low.sum())
            fields[name] = np.where(low, floor, fields[name])
    return clipped


def cell_volume(grid, mxg):
    """
    J dx dy on physical cells, zero in the x boundary cells
    """
    vol = np.asarray(grid["J"]) * np.asarray(grid["dx"]) * np.asarray(grid["dy"])
    vol = vol.copy()
    vol[:mxg] = 0.0
    vol[vol.shape[0] - mxg :] = 0.0
    return vol


def inventories(fields, species, volume, mask=None):
    """
    Particle and energy content of every species: sum N dV, sum 3/2 P dV
    """
    w = volume if mask is None else volume * mask
    w = w[:, :, None]
    out = {"volume": float(w.sum())}
    for s, f in species.items():
        if f["N"]:
            out["N_" + s] = float((fields[f["N"]] * w).sum())
        if f["P"]:
            out["E_" + s] = float((1.5 * fields[f["P"]] * w).sum())
    return out


def conserve(fields, species, target, volume, mask=None):
    """
    Rescale N (and NV with it) and P so that inventories match target.
    Electron energy is rescaled on its own; electron density follows the ions
    """
    current = inventories(fields, species, volume, mask)
    m = np.ones_like(volume) if mask is None else mask
    m = m[:, :, None].astype(bool)
    factors = {}
    ions = has_ions(species)
    ne_before = electron_density(fields, species) if ions else None
    for s, f in species.items():
        if f["N"] and current["N_" + s] != 0:
            a = target["N_" + s] / current["N_" + s]
            fields[f["N"]] = np.where(m, fields[f["N"]] * a, fields[f["N"]])
            if f["NV"]:
                fields[f["NV"]] = np.where(m, fields[f["NV"]] * a, fields[f["NV"]])
            factors["N_" + s] = a
        if f["P"] and current["E_" + s] != 0:
            b = target["E_" + s] / current["E_" + s]
            fields[f["P"]] = np.where(m, fields[f["P"]] * b, fields[f["P"]])
            factors["E_" + s] = b
    electrons = species.get("e")
    if electrons and electrons["NV"] and ions:
        ratio = electron_density(fields, species) / ne_before
        fields[electrons["NV"]] = fields[electrons["NV"]] * ratio
    return factors
