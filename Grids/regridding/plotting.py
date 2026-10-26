"""Plots of fields on the cell polygons of a grid"""

import numpy as np


def cell_polygons(grid):
    """
    Cell corners from the raw gridue arrays rm, zm in the grid file

    Returns
    -------
    polygons : list of [5, 2] arrays (R, Z), corners 1, 2, 4, 3, 1
    index : int array [ncells, 2], (x, y) BOUT index of every polygon
    """
    rm = np.asarray(grid["rm"])
    zm = np.asarray(grid["zm"])
    ny = int(grid["ny"])
    ny_inner = int(grid["ny_inner"])
    npol, nrad = rm.shape[:2]
    cut = (npol - 2) == ny + 2
    polygons = []
    index = []
    for i in range(1, npol - 1):
        p = i - 1
        if cut:
            if p in (ny_inner, ny_inner + 1):
                continue
            j = p if p < ny_inner else p - 2
        else:
            j = p
        for r in range(1, nrad - 1):
            corners = [1, 2, 4, 3, 1]
            polygons.append(np.stack([rm[i, r, corners], zm[i, r, corners]], axis=1))
            index.append((r + 1, j))
    return polygons, np.array(index)


def plot_comparison(old_grid, new_grid, old_fields, new_fields, names=None, zindex=0):
    """
    Old and new fields side by side, one row per field, on the cell
    polygons of each grid
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Polygon

    names = list(new_fields) if names is None else names
    old_cells = cell_polygons(old_grid)
    new_cells = cell_polygons(new_grid)
    fig, axes = plt.subplots(len(names), 2, figsize=(10, 4.5 * len(names)), squeeze=False)
    for row, name in zip(axes, names):
        old = np.asarray(old_fields[name])[..., zindex]
        new = np.asarray(new_fields[name])[..., zindex]
        values = np.concatenate([old.ravel(), new.ravel()])
        if (values > 0).all():
            norm = LogNorm(values.min(), values.max())
        else:
            norm = Normalize(values.min(), values.max())
        for ax, (polygons, index), data, title in (
            (row[0], old_cells, old, "old"),
            (row[1], new_cells, new, "new"),
        ):
            collection = PatchCollection([Polygon(p) for p in polygons], norm=norm, edgecolor="none")
            collection.set_array(data[index[:, 0], index[:, 1]])
            ax.add_collection(collection)
            ax.autoscale_view()
            ax.set_aspect("equal")
            ax.set_title(f"{name} ({title})")
        fig.colorbar(collection, ax=list(row))
    return fig
