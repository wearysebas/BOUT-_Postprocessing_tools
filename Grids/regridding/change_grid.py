"""Conversion of a set of restart files from one grid to another"""

import glob
import os

import numpy as np

from boutdata.collect import collect
from boututils.datafile import DataFile

from .mapping import regrid_fields
from .restart_files import read_restart_scalars, write_restart


def load_grid(filename):
    """
    All variables of a grid file, read once into a dict
    """
    with DataFile(filename) as f:
        return {name: f[name] for name in f.keys()}


def read_restart_fields(path):
    """
    Every (x, y, z) variable of the restart files in path, as global arrays
    including the x boundary cells and without the y guard cells
    """
    with DataFile(os.path.join(path, "BOUT.restart.0.nc")) as f:
        names = [name for name in f.keys() if f.dimensions(name) == ("x", "y", "z")]
    return {
        name: np.asarray(
            collect(
                name,
                path=path,
                xguards=True,
                yguards=False,
                prefix="BOUT.restart",
                info=False,
            )
        )
        for name in names
    }


def change_grid(
    from_grid_file,
    to_grid_file,
    path="data",
    output=".",
    nxpe=None,
    nype=None,
    floors=None,
    conserve_scope="global",
    conserve_mode="adaptive",
    volume_threshold=0.05,
    lambda0=0.0,
    width_pol=None,
    overwrite=False,
    show=False,
):
    """
    Convert a set of restart files from one grid to another, for any
    pair of BOUT++ topologies (closed field line, single null, connected
    and unconnected double null, and the snowflake family)

    Fields are mapped along field lines:
    - radially by the midplane-equivalent distance from the separatrix
    - on closed field lines by the poloidal angle
    - on open field lines within the family of the nearest matching
      target, upstream by the poloidal distance from the X-point and in
      the divertor legs by the fraction of the parallel length to the
      target, blended by a tanh across the X-point
    Densities and temperatures are mapped in log, velocities as NV / N.
    Particle and energy content are then rescaled and floors applied

    Both grid files need a 'topology' variable

    Parameters
    ----------
    from_grid_file : str
        Grid file of the existing simulation
    to_grid_file : str
        Grid file of the new simulation
    path : str, optional
        Directory containing the input restart files
    output : str, optional
        Directory where the new restart files are written
    nxpe, nype : int, optional
        Processor layout of the new restart files. Default: the old one
    floors : dict, optional
        Floor of every density and pressure field. Default: half the
        smallest positive old value
    conserve_scope : str or None, optional
        "global", "closed/open" (closed and open field lines separately)
        or None (no rescaling)
    conserve_mode : str, optional
        "adaptive" (totals if the volume changes by less than
        volume_threshold, averages otherwise), "totals" or "averages"
    volume_threshold : float, optional
        Relative volume change at which "adaptive" switches to averages
    lambda0 : float, optional
        Centre of the tanh blending the upstream and leg maps, as the
        poloidal distance from the X-point [m]
    width_pol : float, optional
        Width of that tanh [m]. Default: 3 poloidal cells at the X-point
        of the new grid
    overwrite : bool, optional
        Replace existing restart files in output
    show : bool, optional
        Plot the old and new fields

    Returns
    -------
    report : dict with inventories, conservation mode and factors, and the
        number of cells raised to the floors
    """
    if not glob.glob(os.path.join(path, "BOUT.restart.*.nc")):
        raise ValueError(f"No restart files found in {path}")

    from_grid = load_grid(from_grid_file)
    to_grid = load_grid(to_grid_file)
    scalars = read_restart_scalars(path)
    fields = read_restart_fields(path)

    new_fields, report = regrid_fields(
        from_grid,
        to_grid,
        fields,
        mxg=int(scalars["MXG"]),
        lambda0=lambda0,
        width_pol=width_pol,
        floors=floors,
        conserve_scope=conserve_scope,
        conserve_mode=conserve_mode,
        volume_threshold=volume_threshold,
    )

    write_restart(output, new_fields, to_grid, scalars, nxpe=nxpe, nype=nype, overwrite=overwrite)

    inventory_old = report["inventory_old"]
    inventory_new = report["inventory_new"]
    print(f"Regridded {len(fields)} fields: {', '.join(fields)}")
    print(f"\tVolume {inventory_old['volume']:.4g} -> {inventory_new['volume']:.4g}")
    for key in inventory_old:
        if key != "volume":
            print(f"\t{key}: {inventory_old[key]:.4g} -> {inventory_new[key]:.4g}")
    if "conserve_mode" in report:
        print(f"\tConservation: {report['conserve_mode']}")
    clipped = {name: n for name, n in report["clipped"].items() if n}
    if clipped:
        print(f"\tCells raised to the floors: {clipped}")

    if show:
        import matplotlib.pyplot as plt

        from .plotting import plot_comparison

        plot_comparison(from_grid, to_grid, fields, new_fields)
        plt.show()

    return report
