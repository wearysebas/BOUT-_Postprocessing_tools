# boutdata.regridding

Restart a BOUT++ simulation on a different grid.

This package converts a set of BOUT++ restart files (`BOUT.restart.*.nc`) from one grid to another. The two grids may differ in resolution, in the position of their points (for example, grids built from different equilibria), or in their magnetic topology. All topologies supported by BoutMesh are handled: closed field line (CFL), single null (SN), unconnected double null (UDN), connected double null (CDN) and the snowflake family (SF+ and SF-, on the low and high field sides).

The mapping follows the magnetic field lines of each grid rather than the grid indices, so that the plasma state of the old simulation is placed on the flux surfaces and field line positions that correspond to it physically on the new grid.

## Requirements

* The `boutdata` and `boututils` packages, `numpy`, and `matplotlib` for the optional plots.
* Both grid files must contain a `topology` variable, as written by the INGRID grid converter (for example `SN`, `SF45`, `SF75`, `SF105`, `SF135`, `SF15`, `SF165`, `UDN`, `CDN`, `CFL`). A grid without this variable, or with a value that is not recognised, is rejected with an error. A bare `SF` value is accepted and treated as a snowflake+ on the low field side, as BoutMesh does, with a warning.
* The grid files must contain the geometry of the cells (`Rxy`, `Zxy`, `psixy`, `Bxy`, `Bpxy`, `hthe`, `dx`, `dy`, `J`), and `rm`, `zm` for the optional plots.
* The grids must have closed field lines, which are used to locate the separatrix.

## Quick start

```python
from boutdata.regridding import change_grid

report = change_grid(
    "old/old_bout.grd.nc",   # grid of the existing simulation
    "new/new_bout.grd.nc",   # grid of the new simulation
    path="old/run",                 # directory with BOUT.restart.*.nc
    output="new/run",               # directory for the new restart files
)
```

The new restart files are written to `output` with the processor layout of the old simulation. To start the new simulation from them:

1. Place the new grid file in the location given by `mesh:file` in the `BOUT.inp` of the new run, and check that this option refers to the new grid.
2. Run on the number of processors of the new restart files (`NXPE * NYPE`).
3. Pass `restart` on the command line, for example `mpirun -np 9 ./hermes-3 -d new/run restart`. Without this argument BOUT++ ignores the restart files and starts from the initial conditions in `BOUT.inp`.

The simulation time continues from the time stored in the old restart files.

## Parameters of `change_grid`

| Parameter | Default | Description |
|---|---|---|
| `from_grid_file` | | Grid file of the existing simulation. |
| `to_grid_file` | | Grid file of the new simulation. |
| `path` | `"data"` | Directory containing the input restart files. |
| `output` | `"."` | Directory where the new restart files are written. |
| `nxpe`, `nype` | old layout | Processor layout of the new restart files. |
| `floors` | half the smallest positive old value | Floor applied to every density and pressure field, as a dictionary `{field name: value}`. |
| `conserve_scope` | `"global"` | `"global"` rescales the whole domain, `"closed/open"` rescales closed and open field lines separately, `None` disables the rescaling. |
| `conserve_mode` | `"adaptive"` | `"totals"` preserves the particle and energy content, `"averages"` preserves them per unit volume, `"adaptive"` uses totals when the plasma volume changes by less than `volume_threshold` and averages otherwise. |
| `volume_threshold` | `0.05` | Relative volume change at which the adaptive mode changes from totals to averages. |
| `lambda0` | `0.0` | Centre of the transition between the upstream and divertor maps, as a poloidal distance from the X-point in metres. |
| `width_pol` | 3 poloidal cells | Width of that transition in metres. By default it is three times the median poloidal cell length at the X-point of the new grid. |
| `overwrite` | `False` | Allow replacing existing restart files in `output`. |
| `show` | `False` | Plot the old and new fields on the cell polygons of each grid. |

The function returns a report (a dictionary) containing the particle and energy content of each species before and after the conversion, the plasma volume of both grids, the conservation mode applied, the rescaling factors, and the number of cells raised to the floors. A summary is also printed.

## Processor layout

The processor layout of the new restart files must be accepted by BoutMesh for the new grid. Before writing, the layout is checked with the same conditions as `bout::checkBoutMeshYDecomposition` and `BoutMesh::topology()`. If the old layout is not valid on the new grid, an error is raised that lists the nearest valid processor counts, and `nxpe` and `nype` must then be given explicitly. The valid layouts of a grid can also be listed directly:

```python
from boutdata.regridding import load_grid, valid_decompositions

grid = load_grid("new/grid.nc")
valid_decompositions(grid, npes=36, mxg=2, myg=2)   # list of valid (NXPE, NYPE)
```

## Method

### Topology

The topology is identified in the same way as BoutMesh: the `topology` variable gives the family, and for snowflakes the number in it gives the type (105 and 135: SF+ on the high field side; 165: SF- on the high field side; 15: SF- on the low field side; 45 and 75: SF+ on the low field side). The connections between the regions of the grid are then reproduced from the `set_connection` and `add_target` calls of `BoutMesh::topology()`, which gives, for every cell, its neighbours along the magnetic field. Following these neighbours separates closed from open field lines and identifies the target at each end of every open field line.

### Coordinates

* Radial: the midplane-equivalent distance from the separatrix, `rho = (psi - psi_sep) / (R Bp)`, evaluated at the separatrix on the outboard midplane. Near the separatrix it equals the radial distance from the separatrix at the outboard midplane, so upstream profiles and decay lengths are preserved between grids built from different equilibria.
* Closed field lines: the poloidal angle about the magnetic axis.
* Open field lines: each field line is divided at its poloidal midpoint, and each half belongs to its nearest target. Targets of the new grid are matched to the nearest targets of the old grid in the poloidal plane. Along each half, two coordinates are used: the poloidal distance from the X-point, `lambda`, and the fraction of the parallel length to the target, `ell_T`.

### Interpolation

Every field line of the old grid has a single value of `rho`. A cell of the new grid takes its value from the two old field lines, of the same family, whose `rho` brackets its own: each of them is interpolated in one dimension along its coordinate, and the two results are combined linearly in `rho`. Outside the radial range of the old grid, the value of the nearest old field line is used. This interpolation reproduces the input exactly when both grids are the same, and it never mixes field lines that end on different targets.

On open field lines two maps are computed: an upstream map in `(rho, lambda)`, which places points at the same distance from the X-point, and a divertor map in `(rho, ell_T)`, which places points at the same fraction of the way to the target. They are blended with

`w = 0.5 * (1 + tanh((lambda - lambda0) / width_pol))`,
`f = w * f_upstream + (1 - w) * f_divertor`,

so that the upstream map is used above the X-point and the divertor map along the legs. On closed field lines `w = 1`.

### Physical quantities

Fields are grouped by species from the Hermes-3 naming convention (`N<species>`, `P<species>`, `NV<species>`). The quantities interpolated are the logarithm of the density, the logarithm of the temperature `P / N`, and the velocity `NV / N`; the pressure and the momentum are rebuilt from them on the new grid. This preserves positivity and avoids spurious temperatures. The electron density is taken from quasineutrality, as the sum of the ion densities weighted by their charge. When no ion density is evolved (for example with a fixed electron density), the logarithm of the pressure is interpolated directly. Other fields, such as potentials, are interpolated as they are.

After the mapping, the particle and energy content of each species is rescaled according to `conserve_scope` and `conserve_mode`, and the floors are applied. Only the physical cells are used as sources; the x boundary cells of the old restart files are not.

### Output

All scalar variables of the old restart files are copied, including the simulation time, the iteration counter and the normalisations. The variables that describe the grid and the processor layout are replaced by those of the new grid. The topology is recorded as BoutMesh records it: `mesh_topology` (for example `snowflake` or `single_null`), `IngridTopology` (the value of the `topology` variable) and, for snowflakes only, `snowflake_type`. The two dimensional metrics of the new grid are written with the fields. The y guard cells are written as zero and are filled by BOUT++ at the first communication.

## Lower level functions

| Function | Module | Purpose |
|---|---|---|
| `regrid_fields(old_grid, new_grid, fields, ...)` | `mapping` | Maps a dictionary of global arrays `[nx, ny, nz]` between two grids given as dictionaries, without reading or writing files. |
| `write_restart(output, fields, grid, scalars, ...)` | `restart_files` | Splits global arrays into restart files for a grid and a processor layout. |
| `read_topology(grid)` | `topology` | Returns the topology family and, for snowflakes, the snowflake type. |
| `topology_regions(grid)` | `topology` | Divides a grid into logically rectangular regions with their connections and targets. |
| `load_grid(filename)`, `read_restart_fields(path)` | `change_grid` | Read a grid file or the evolved fields of a set of restart files. |
| `plot_comparison(old_grid, new_grid, old_fields, new_fields)` | `plotting` | Plots old and new fields on the cell polygons of each grid. |

## Package layout

| Module | Content |
|---|---|
| `topology.py` | Topology identification, field line connectivity and regions, following BoutMesh. |
| `coordinates.py` | Radial and along field line coordinates, field line pieces, target matching. |
| `physics.py` | Species grouping, interpolated quantities, floors, particle and energy content, rescaling. |
| `mapping.py` | Interpolation along field lines, blending of the upstream and divertor maps, `regrid_fields`. |
| `restart_files.py` | Processor layout checks and writing of restart files. |
| `change_grid.py` | The `change_grid` entry point. |
| `plotting.py` | Plots on the cell polygons of a grid. |

## Limitations

* The X-point target topology is not supported, since BoutMesh does not define its connections.
* Grids without closed field lines are not supported.
* The mapping is axisymmetric: for three dimensional fields, every toroidal slice is mapped independently with the same coefficients.
* Grid cells beyond the radial range of the old grid take the values of the outermost old field lines. The new simulation relaxes these cells from that state.
* Fields other than the evolved variables (for example `G1`, `G2`, `G3`) are not written, since BOUT++ computes them and does not read them from restart files.

## Tests

The tests are in `boutdata/tests/test_regridding.py` and use synthetic grids with the index layouts of every supported topology:

```
pytest src/boutdata/tests/test_regridding.py
```

Additional tests on real grids and simulations run when the environment variable `BOUT_REGRID_TESTS` points to a directory containing a `Tagged_grids` folder and the corresponding restart files; otherwise they are skipped.
