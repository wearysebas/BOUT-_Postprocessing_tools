#!/usr/bin/env python3
"""
BOUT++ 3D visualization of the full reactor using gridue cell corners.

Sweeps the 2D poloidal cross-section toroidally to build a 3D polygon
representation of the tokamak, colored by simulation data at each
toroidal slice.

Usage:
    python BOUT_3D_Postprocessing.py <grid_file> <sim_results_dir> [options]

Example:
    python BOUT_3D_Postprocessing.py \
        1st_try/gridue_try6_bout_from_in.grd.nc 1st_try \
        --var T --time -1 --nslices 32
"""

import numpy as np
import xarray as xr
import xbout
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def build_cell_mapping(rm, zm, ny_inner, bout_nx, bout_ny):
    """
    Build the list of gridue cells with their 2D corner coordinates
    and corresponding BOUT grid indices.

    Returns a list of (r_corners, z_corners, bout_i, bout_j) tuples,
    where r/z_corners are 1D arrays of 5 values (closed quadrilateral).
    """
    Nx_gridue = rm.shape[0]
    Ny_gridue = rm.shape[1]
    idx = [np.array([1, 2, 4, 3, 1])]

    cells = []
    for i in range(Nx_gridue):
        for j in range(Ny_gridue):
            if j == 0 or j == Ny_gridue - 1:
                continue
            bout_i = j + 1

            if i == 0 or i == Nx_gridue - 1:
                continue
            post_step1 = i - 1
            if post_step1 < ny_inner:
                bout_j = post_step1
            elif post_step1 == ny_inner or post_step1 == ny_inner + 1:
                continue
            else:
                bout_j = post_step1 - 2

            if bout_i < 0 or bout_i >= bout_nx or bout_j < 0 or bout_j >= bout_ny:
                continue

            r_corners = rm[i][j][idx].flatten()
            z_corners = zm[i][j][idx].flatten()
            cells.append((r_corners, z_corners, bout_i, bout_j))

    return cells


def build_boundary_edges(rm, zm, ny_inner, bout_nx, bout_ny):
    """
    Collect boundary edges of the grid for building the torus skin.

    Returns a list of (R_a, Z_a, R_b, Z_b, bout_i, bout_j) tuples
    for the outer radial, inner radial, and poloidal boundary edges.

    Gridue corner layout:
        (1) -- (3)
         |      |         ^
         |  (0) |  → Radial (j)
         |      |         |
        (2) -- (4)    Poloidal (i)

    Outer radial edge: corners 3 → 4
    Inner radial edge: corners 1 → 2
    Upper poloidal edge: corners 1 → 3
    Lower poloidal edge: corners 2 → 4
    """
    Nx_gridue = rm.shape[0]
    Ny_gridue = rm.shape[1]

    edges = []

    # Helper to convert gridue poloidal index i to bout_j
    def gridue_i_to_bout_j(i):
        if i == 0 or i == Nx_gridue - 1:
            return None
        ps = i - 1
        if ps == ny_inner or ps == ny_inner + 1:
            return None
        return ps if ps < ny_inner else ps - 2

    # Valid poloidal indices (for iterating)
    valid_i = [i for i in range(1, Nx_gridue - 1)
               if gridue_i_to_bout_j(i) is not None]

    # --- Outer radial boundary (j = Ny_gridue - 2): corners 3 → 4 ---
    j_outer = Ny_gridue - 2
    bout_i_outer = j_outer + 1
    if bout_i_outer < bout_nx:
        for i in valid_i:
            bout_j = gridue_i_to_bout_j(i)
            if bout_j is not None and bout_j < bout_ny:
                edges.append((rm[i][j_outer][3], zm[i][j_outer][3],
                              rm[i][j_outer][4], zm[i][j_outer][4],
                              bout_i_outer, bout_j))

    # --- Inner radial boundary (j = 1): corners 1 → 2 ---
    j_inner = 1
    bout_i_inner = j_inner + 1
    if bout_i_inner < bout_nx:
        for i in valid_i:
            bout_j = gridue_i_to_bout_j(i)
            if bout_j is not None and bout_j < bout_ny:
                edges.append((rm[i][j_inner][1], zm[i][j_inner][1],
                              rm[i][j_inner][2], zm[i][j_inner][2],
                              bout_i_inner, bout_j))

    # --- Poloidal boundaries: top/bottom edges at leg endpoints ---
    # Indices where poloidal range starts/ends (first valid, last valid,
    # and around guard cells at ny_inner)
    pol_starts = []  # indices with exposed bottom edge (2 → 4)
    pol_ends = []    # indices with exposed top edge (1 → 3)

    for idx_pos, i in enumerate(valid_i):
        # Bottom edge exposed if previous i is not valid
        if idx_pos == 0 or valid_i[idx_pos - 1] != i - 1:
            pol_starts.append(i)
        # Top edge exposed if next i is not valid
        if idx_pos == len(valid_i) - 1 or valid_i[idx_pos + 1] != i + 1:
            pol_ends.append(i)

    for j in range(1, Ny_gridue - 1):
        bout_i = j + 1
        if bout_i >= bout_nx:
            continue
        for i in pol_starts:
            bout_j = gridue_i_to_bout_j(i)
            if bout_j is not None and bout_j < bout_ny:
                edges.append((rm[i][j][2], zm[i][j][2],
                              rm[i][j][4], zm[i][j][4],
                              bout_i, bout_j))
        for i in pol_ends:
            bout_j = gridue_i_to_bout_j(i)
            if bout_j is not None and bout_j < bout_ny:
                edges.append((rm[i][j][1], zm[i][j][1],
                              rm[i][j][3], zm[i][j][3],
                              bout_i, bout_j))

    return edges


def plot_torus_slices(ax, all_r, all_z, all_bi, all_bj, data_3d,
                      z_values, nslices, toroidal_range, cmap, var_label):
    """
    Render the torus as opaque cross-section slices (cutaway mode).
    """
    ncells = len(all_r)
    max_angle = np.radians(toroidal_range)
    toroidal_angles = np.linspace(0, max_angle, nslices, endpoint=False)
    z_indices = np.array([np.argmin(np.abs(z_values - a % (2 * np.pi)))
                          for a in toroidal_angles])

    norm = Normalize(vmin=np.nanmin(data_3d), vmax=np.nanmax(data_3d))
    colormap = plt.get_cmap(cmap)

    print(f"Building 3D mesh: {ncells} cells x {nslices} slices "
          f"= {ncells * nslices} polygons...")

    verts = []
    face_colors = []

    for zeta, z_idx in zip(toroidal_angles, z_indices):
        cos_z, sin_z = np.cos(zeta), np.sin(zeta)
        x_3d = all_r * cos_z
        y_3d = all_r * sin_z
        values = data_3d[all_bi, all_bj, z_idx]
        colors = colormap(norm(values))

        for c in range(ncells):
            verts.append(list(zip(x_3d[c], y_3d[c], all_z[c])))
        face_colors.extend(colors)

    poly = Poly3DCollection(verts, facecolors=face_colors,
                            edgecolors=face_colors, linewidths=0.1)
    ax.add_collection3d(poly)
    return norm, colormap


def plot_torus_with_focus(ax, all_r, all_z, all_bi, all_bj, data_3d,
                          z_values, nslices, focus_zindex, cmap,
                          var_label, bg_alpha=0.15,
                          boundary_edges=None):
    """
    Render the full torus with translucent connected surface and one
    opaque focus slice.

    The torus skin is built by connecting adjacent toroidal slices with
    quad faces along the radial and poloidal boundaries, rendered at low
    opacity.  A single toroidal cross-section is drawn at full opacity.

    Parameters
    ----------
    focus_zindex : int
        Simulation z-index of the slice to highlight.
    bg_alpha : float
        Opacity of the background torus surface (0-1, default 0.15).
    boundary_edges : list, optional
        Output of build_boundary_edges().  If None, no connecting skin
        is drawn and only cross-section faces are rendered.
    """
    ncells = len(all_r)
    toroidal_angles = np.linspace(0, 2 * np.pi, nslices, endpoint=False)
    z_indices = np.array([np.argmin(np.abs(z_values - a % (2 * np.pi)))
                          for a in toroidal_angles])

    # Find which rendering slice is closest to the requested focus z-index
    focus_angle = z_values[focus_zindex]
    focus_render_idx = int(np.argmin(np.abs(toroidal_angles - focus_angle)))

    norm = Normalize(vmin=np.nanmin(data_3d), vmax=np.nanmax(data_3d))
    colormap = plt.get_cmap(cmap)

    # ---- Cross-section faces (focus = opaque, rest = translucent) ----
    bg_verts = []
    bg_colors = []
    focus_verts = []
    focus_colors = []

    for k, (zeta, z_idx) in enumerate(zip(toroidal_angles, z_indices)):
        cos_z, sin_z = np.cos(zeta), np.sin(zeta)
        x_3d = all_r * cos_z
        y_3d = all_r * sin_z

        values = data_3d[all_bi, all_bj, z_idx]
        colors = colormap(norm(values))  # (ncells, 4) RGBA

        is_focus = (k == focus_render_idx)

        for c in range(ncells):
            v = list(zip(x_3d[c], y_3d[c], all_z[c]))
            if is_focus:
                focus_verts.append(v)
                focus_colors.append(colors[c])
            else:
                bg_verts.append(v)
                rgba = np.array(colors[c], dtype=float)
                rgba[3] = bg_alpha
                bg_colors.append(rgba)

    # ---- Connecting skin faces between adjacent slices ----
    if boundary_edges is not None:
        n_edges = len(boundary_edges)
        # Pre-extract boundary edge arrays for vectorised operations
        edge_Ra = np.array([e[0] for e in boundary_edges])
        edge_Za = np.array([e[1] for e in boundary_edges])
        edge_Rb = np.array([e[2] for e in boundary_edges])
        edge_Zb = np.array([e[3] for e in boundary_edges])
        edge_bi = np.array([e[4] for e in boundary_edges])
        edge_bj = np.array([e[5] for e in boundary_edges])

        for k in range(nslices):
            k_next = (k + 1) % nslices
            zeta_k = toroidal_angles[k]
            zeta_next = toroidal_angles[k_next]
            z_idx = z_indices[k]

            cos_k, sin_k = np.cos(zeta_k), np.sin(zeta_k)
            cos_n, sin_n = np.cos(zeta_next), np.sin(zeta_next)

            values = data_3d[edge_bi, edge_bj, z_idx]
            colors = colormap(norm(values))

            for e in range(n_edges):
                Ra, Za = edge_Ra[e], edge_Za[e]
                Rb, Zb = edge_Rb[e], edge_Zb[e]
                quad = [
                    (Ra * cos_k, Ra * sin_k, Za),
                    (Rb * cos_k, Rb * sin_k, Zb),
                    (Rb * cos_n, Rb * sin_n, Zb),
                    (Ra * cos_n, Ra * sin_n, Za),
                ]
                bg_verts.append(quad)
                rgba = np.array(colors[e], dtype=float)
                rgba[3] = bg_alpha
                bg_colors.append(rgba)

        print(f"Building 3D mesh: {ncells} cells x {nslices} slices "
              f"+ {n_edges * nslices} skin faces  "
              f"(focus slice {focus_render_idx})")
    else:
        print(f"Building 3D mesh: {ncells} cells x {nslices} slices "
              f"(focus slice {focus_render_idx})")

    # Background torus (translucent cross-sections + skin)
    if bg_verts:
        bg_poly = Poly3DCollection(bg_verts, facecolors=bg_colors,
                                   edgecolors="none", linewidths=0)
        ax.add_collection3d(bg_poly)

    # Focus slice (opaque, with thin edges for cell visibility)
    if focus_verts:
        focus_poly = Poly3DCollection(focus_verts, facecolors=focus_colors,
                                      edgecolors=focus_colors, linewidths=0.1)
        ax.add_collection3d(focus_poly)

    return norm, colormap


def setup_axes(ax, rm, zm, elev, azim):
    """Configure 3D axis limits, labels, and camera."""
    R_max = rm[:, :, 0].max() * 1.05
    Z_min = zm[:, :, 0].min()
    Z_max = zm[:, :, 0].max()
    ax.set_xlim(-R_max, R_max)
    ax.set_ylim(-R_max, R_max)
    ax.set_zlim(Z_min, Z_max)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_box_aspect([1, 1, (Z_max - Z_min) / (2 * R_max)])
    ax.view_init(elev=elev, azim=azim)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BOUT++ 3D polygon visualization of the full reactor"
    )
    parser.add_argument("grid_file", type=str, help="Path to grid file (.nc)")
    parser.add_argument("sim_results", type=str,
                        help="Path to simulation results directory")
    parser.add_argument("--var", type=str, default="T",
                        help="Variable to plot (default: T)")
    parser.add_argument("--time", type=int, default=-1,
                        help="Time index (default: -1 = last)")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Colormap (default: viridis)")
    parser.add_argument("--nslices", type=int, default=32,
                        help="Number of toroidal slices to render (default: 32)")
    parser.add_argument("--toroidal_range", type=float, default=360.0,
                        help="Toroidal range in degrees for cutaway view (default: 360)")
    parser.add_argument("--zindex", type=int, default=None,
                        help="Focus on this toroidal slice (enables focus mode)")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Background torus opacity in focus mode (default: 0.15)")
    parser.add_argument("--elev", type=float, default=20.0,
                        help="Camera elevation angle in degrees (default: 20)")
    parser.add_argument("--azim", type=float, default=45.0,
                        help="Camera azimuth angle in degrees (default: 45)")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    # Load grid file and simulation results
    grid = xr.open_dataset(args.grid_file, engine="netcdf4")
    sim = xbout.open_boutdataset(args.sim_results + "/BOUT.dmp.*.nc")

    MXG = sim.metadata["MXG"]
    nx = int(grid["nx"].values)
    ny = int(grid["ny"].values)
    ny_inner = int(grid["ny_inner"].values)

    # Full 3D data for this time step: (bout_nx, bout_ny, nz)
    data_3d = sim[args.var].values[args.time, MXG:-MXG, :, :]
    nz = data_3d.shape[2]
    z_values = sim["z"].values  # toroidal angles in radians

    rm = grid["rm"].values
    zm = grid["zm"].values

    # Build cell mapping once
    cells = build_cell_mapping(rm, zm, ny_inner, nx, ny)
    all_r = np.array([c[0] for c in cells])
    all_z = np.array([c[1] for c in cells])
    all_bi = np.array([c[2] for c in cells])
    all_bj = np.array([c[3] for c in cells])

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    if args.zindex is not None:
        # Build boundary edges for the torus skin
        boundary_edges = build_boundary_edges(rm, zm, ny_inner, nx, ny)

        # Focus mode: full torus translucent + one opaque slice
        norm, colormap = plot_torus_with_focus(
            ax, all_r, all_z, all_bi, all_bj, data_3d, z_values,
            args.nslices, args.zindex, args.cmap, args.var, args.alpha,
            boundary_edges=boundary_edges,
        )
    else:
        # Cutaway mode: opaque slices over toroidal range
        norm, colormap = plot_torus_slices(
            ax, all_r, all_z, all_bi, all_bj, data_3d, z_values,
            args.nslices, args.toroidal_range, args.cmap, args.var,
        )

    setup_axes(ax, rm, zm, args.elev, args.azim)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=args.var, shrink=0.6, pad=0.1)

    times = sim["t"].values
    t_val = times[args.time]
    ax.set_title(f"{args.var}  t = {t_val:.3f}")

    plt.tight_layout()
    plt.show()
