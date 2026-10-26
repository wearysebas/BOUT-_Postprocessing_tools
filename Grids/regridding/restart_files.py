"""Reading and writing BOUT++ restart files for a new grid"""

import glob
import os
import pathlib

import numpy as np

from boututils.datafile import DataFile

from .topology import read_topology, y_decomposition_indices


MESH_TOPOLOGY_NAMES = {
    "CFL": "closed_field_line",
    "SN": "single_null",
    "UDN": "unconnected_double_null",
    "CDN": "connected_double_null",
    "SF": "snowflake",
    "XPT": "XPoint_target",
}


SNOWFLAKE_TYPE_NAMES = {
    "SF_minus_LFS": "SF_minus_low_field_side",
    "SF_minus_HFS": "SF_minus_high_field_side",
    "SF_plus_LFS": "SF_plus_low_field_side",
    "SF_plus_HFS": "SF_plus_high_field_side",
    "SF": "SF",
    "XPT": "XPT",
}


GRID_SCALARS = (
    "nx",
    "ny",
    "ixseps1",
    "ixseps2",
    "jyseps1_1",
    "jyseps2_1",
    "jyseps1_2",
    "jyseps2_2",
    "ny_inner",
)


GRID_METRICS = (
    "dx",
    "dy",
    "J",
    "Bxy",
    "g11",
    "g22",
    "g33",
    "g12",
    "g13",
    "g23",
    "g_11",
    "g_22",
    "g_33",
    "g_12",
    "g_13",
    "g_23",
    "zShift",
)


def check_y_decomposition(grid, nype, myg):
    """
    Port of bout::checkBoutMeshYDecomposition and the MYSUB checks in
    BoutMesh::topology()

    Returns
    -------
    (ok, message)
    """
    family, sf_type = read_topology(grid)
    ind = y_decomposition_indices(grid)
    jyseps1_1 = ind["jyseps1_1"]
    jyseps2_1 = ind["jyseps2_1"]
    jyseps1_2 = ind["jyseps1_2"]
    jyseps2_2 = ind["jyseps2_2"]
    ny_inner = ind["ny_inner"]
    ny = int(grid["ny"])

    if ny % nype != 0:
        return False, f"ny ({ny}) must be divisible by NYPE ({nype})"
    mysub = ny // nype
    if mysub < myg:
        return False, f"ny/NYPE ({ny}/{nype} = {mysub}) must be >= MYG ({myg})"

    checks = [("jyseps1_1+1", jyseps1_1 + 1)]
    if family in ("UDN", "CDN"):
        checks += [
            ("jyseps2_1-jyseps1_1", jyseps2_1 - jyseps1_1),
            ("jyseps2_2-jyseps1_2", jyseps2_2 - jyseps1_2),
            ("ny_inner-jyseps2_1-1", ny_inner - jyseps2_1 - 1),
            ("jyseps1_2-ny_inner+1", jyseps1_2 - ny_inner + 1),
        ]
    elif family == "SF" and sf_type in ("SF_plus_LFS", "SF"):
        checks += [
            ("jyseps2_1-jyseps1_1", jyseps2_1 - jyseps1_1),
            ("jyseps2_2-ny_inner+1", jyseps2_2 - ny_inner + 1),
            ("ny_inner-1-jyseps1_2", ny_inner - 1 - jyseps1_2),
            ("jyseps1_2-jyseps2_1", jyseps1_2 - jyseps2_1),
            ("ny-1-jyseps2_2", ny - 1 - jyseps2_2),
            ("ny_inner-1-jyseps2_1", ny_inner - 1 - jyseps2_1),
            ("ny-ny_inner", ny - ny_inner),
        ]
    elif family == "SF" and sf_type == "SF_plus_HFS":
        checks += [
            ("jyseps1_2-jyseps2_1", jyseps1_2 - jyseps2_1),
            ("jyseps2_2-ny_inner+1", jyseps2_2 - ny_inner + 1),
            ("ny_inner-1-jyseps1_2", ny_inner - 1 - jyseps1_2),
            ("jyseps2_1-jyseps1_1", jyseps2_1 - jyseps1_1),
            ("ny-1-jyseps2_2", ny - 1 - jyseps2_2),
            ("jyseps2_1+1", jyseps2_1 + 1),
            ("ny-ny_inner", ny - ny_inner),
            ("ny_inner-jyseps1_2-1", ny_inner - jyseps1_2 - 1),
        ]
    elif family == "SF" and sf_type in ("SF_minus_LFS", "SF_minus_HFS"):
        checks += [
            ("jyseps2_1-jyseps1_1", jyseps2_1 - jyseps1_1),
            ("jyseps2_2-jyseps1_2", jyseps2_2 - jyseps1_2),
            ("jyseps1_2-ny_inner+1", jyseps1_2 - ny_inner + 1),
            ("ny_inner-1-jyseps2_1", ny_inner - 1 - jyseps2_1),
            ("ny-1-jyseps2_2", ny - 1 - jyseps2_2),
            ("jyseps2_1+1", jyseps2_1 + 1),
            ("ny-1-jyseps1_2", ny - 1 - jyseps1_2),
            ("ny-ny_inner", ny - ny_inner),
        ]
        if sf_type == "SF_minus_HFS":
            checks += [("jyseps2_2-jyseps1_1", jyseps2_2 - jyseps1_1)]
    elif family in ("SN", "CFL"):
        checks += [("jyseps2_2-jyseps1_1", jyseps2_2 - jyseps1_1)]
    else:
        return False, f"No y decomposition check for {family} {sf_type}"
    checks += [("ny-1-jyseps2_2", ny - 1 - jyseps2_2)]

    for label, value in checks:
        if value % mysub != 0:
            return False, f"{label} ({value}) must be a multiple of MYSUB ({mysub})"
    return True, ""


def check_x_decomposition(grid, nxpe, mxg):
    """
    Checks on NXPE made in BoutMesh::topology()
    """
    nx = int(grid["nx"])
    mx = nx - 2 * mxg
    if mx % nxpe != 0:
        return False, f"nx - 2*MXG ({mx}) must be divisible by NXPE ({nxpe})"
    if nxpe > 1 and mx // nxpe < mxg:
        return False, f"MXSUB ({mx // nxpe}) must be >= MXG ({mxg})"
    return True, ""


def valid_decompositions(grid, npes, mxg, myg):
    """
    All (NXPE, NYPE) with NXPE * NYPE == npes accepted by BoutMesh
    """
    valid = []
    for nxpe in range(1, npes + 1):
        if npes % nxpe != 0:
            continue
        nype = npes // nxpe
        if check_x_decomposition(grid, nxpe, mxg)[0] and check_y_decomposition(grid, nype, myg)[0]:
            valid.append((nxpe, nype))
    return valid


def read_restart_scalars(path):
    """
    All scalar variables of BOUT.restart.0.nc in path
    """
    with DataFile(os.path.join(path, "BOUT.restart.0.nc")) as f:
        return {k: f[k] for k in f.keys() if f.ndims(k) == 0}


def write_restart(
    output, fields, grid, scalars, nxpe=None, nype=None, overwrite=False
):
    """
    Split global fields [nx, ny, nz] into BOUT.restart.*.nc files for grid

    Parameters
    ----------
    output : str
        Directory to write into
    fields : dict name -> array [nx, ny, nz], including x boundary cells,
        without y guard cells
    grid : DataFile or dict-like
        The new grid
    scalars : dict
        Scalars of the old restart files, copied except for the ones that
        describe the grid and the processor layout
    nxpe, nype : int, optional
        Processor layout. Default is the old one
    overwrite : bool
        Allow replacing existing restart files in output
    """
    output = pathlib.Path(output)
    if not overwrite and glob.glob(str(output / "BOUT.restart.*.nc")):
        raise ValueError(f"{output} already has restart files; use overwrite=True")

    mxg = int(scalars["MXG"])
    myg = int(scalars["MYG"])
    nxpe = int(scalars["NXPE"]) if nxpe is None else nxpe
    nype = int(scalars["NYPE"]) if nype is None else nype

    for ok, message in (
        check_x_decomposition(grid, nxpe, mxg),
        check_y_decomposition(grid, nype, myg),
    ):
        if not ok:
            npes = nxpe * nype
            options = {
                n: valid_decompositions(grid, n, mxg, myg) for n in range(1, 4 * npes + 1)
            }
            options = {n: v for n, v in options.items() if v}
            nearest = sorted(options, key=lambda n: abs(n - npes))[:4]
            raise ValueError(
                f"NXPE={nxpe}, NYPE={nype} not valid on the new grid: {message}. "
                "Nearest valid (NXPE, NYPE): "
                + ", ".join(f"{n} processors {options[n]}" for n in sorted(nearest))
            )

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    mxsub = (nx - 2 * mxg) // nxpe
    mysub = ny // nype
    family, sf_type = read_topology(grid)

    new_scalars = dict(scalars)
    for name in GRID_SCALARS:
        new_scalars[name] = int(grid[name])
    new_scalars["mesh_topology"] = MESH_TOPOLOGY_NAMES[family]
    new_scalars["IngridTopology"] = str(grid["topology"]).strip().upper()
    new_scalars.pop("snowflake_type", None)
    if family == "SF":
        new_scalars["snowflake_type"] = SNOWFLAKE_TYPE_NAMES[sf_type]
    new_scalars["NXPE"] = nxpe
    new_scalars["NYPE"] = nype
    new_scalars["MXSUB"] = mxsub
    new_scalars["MYSUB"] = mysub

    metrics = {
        name: np.asarray(grid[name], dtype=float) for name in GRID_METRICS if name in grid.keys()
    }

    output.mkdir(parents=True, exist_ok=True)
    for i in range(nxpe * nype):
        ix = i % nxpe
        iy = i // nxpe
        xs = slice(ix * mxsub, (ix + 1) * mxsub + 2 * mxg)
        ys = slice(iy * mysub, (iy + 1) * mysub)

        def block(data):
            out = np.zeros((mxsub + 2 * mxg, mysub + 2 * myg) + data.shape[2:])
            out[:, myg : myg + mysub] = data[xs, ys]
            return out

        with DataFile(output / f"BOUT.restart.{i}.nc", create=True, format="NETCDF4") as f:
            for name, value in new_scalars.items():
                f.write(name, value)
            f.write("PE_XIND", ix)
            f.write("PE_YIND", iy)
            f.write("MYPE", i)
            for name, value in metrics.items():
                f.write(name, block(value))
            for name, value in fields.items():
                f.write(name, block(value))
