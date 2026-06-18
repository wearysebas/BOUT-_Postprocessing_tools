import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import xbout

### Visualization tools for BOUT++ grid files. 

# Load the grid file
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="""shows BOUT++ grid files."""
    )
    parser.add_argument("grid_file1", type=str)
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()
    grid_file1 = args.grid_file1


    d1 = xr.open_dataset(grid_file1, engine="netcdf4")




def plot_lines(ds,grid_file,topology):
    """
    Function to plot the grid lines of a BOUT++ grid file.
    
    ds: BOUT++ grid file as an xarray dataset.
    grid_file: The path to the BOUT++ grid file, used for the plot title.
    topology: The topology of the grid, used to determine how to plot the separatrix lines
    """

    # Extract R and Z coordinates
    R = ds["Rxy"].values
    Z = ds["Zxy"].values

    #Extract the indices of the separatrix lines and y branch-cuts
    ixseps1 = ds["ixseps1"].values
    ixseps2 = ds["ixseps2"].values
    jyseps1_1 = ds["jyseps1_1"].values
    jyseps1_2 = ds["jyseps1_2"].values
    ny_inner = ds["ny_inner"].values
    jyseps2_1 = ds["jyseps2_1"].values
    jyseps2_2 = ds["jyseps2_2"].values
    
    print("Topology:", ds["topology"].values)

    #Check the separatrix indices to confirm they are where expected.
    print("ixseps1: ", ixseps1)
    print("ixseps2: ", ixseps2)
    
    #Check the branch cut indices to confirm they are where expected.
    print("jyseps1_1: ", jyseps1_1)
    print("jyseps2_1: ", jyseps2_1)
    print("jyseps1_2: ", jyseps1_2)
    print("jyseps2_2: ", jyseps2_2)
    print("ny_inner: ", ny_inner)

    # Confirm shape
    print("Rxy shape:", R.shape)
    print("Zxy shape:", Z.shape)

    #The following include guard cells.
    nx = R.shape[0] # Number of points in the x-direction
    ny = R.shape[1] # Number of points in the y-direction


    r = ds["rm"].values #3rd coordinate is the position in the cell (0-4), where 0 is center, 1 is left poloidally, 2 is up radially, 3 is right poloidally, 4 is down radially, 3 is down radially, 4 is right poloidally.
    z = ds["zm"].values

    r = r.transpose(1, 0, 2)
    z = z.transpose(1, 0, 2) 

    #ds = xbout.geometries.add_toroidal_geometry_coords(ds)
    #xbout.plotting.plotfuncs.plot2d_polygon(ds["Bxy"], ax = ax)

    ax = plot(ds, show=False)

    if topology == "SF":
        ixseps_sep = int(ny / 9 * 7)  # Approx location of separatrix in x-direction for SF topology
        ixseps_sep2 = int(ny / 9)  # Approx location of separatrix in x-direction for SF topology

        # Core periodic section for ixseps2
        r_core = r[ixseps2-1, ixseps_sep2+1:5*ixseps_sep2+1, 1]
        z_core = z[ixseps2-1, ixseps_sep2+1:5*ixseps_sep2+1, 1]

        # Append the first point again to close the loop
        r_core = np.append(r_core, r_core[0])
        z_core = np.append(z_core, z_core[0])

        # ixseps2 full East PFR section
        r_EPFR = r[ixseps2-1, 0:ixseps_sep2+1,1]
        z_EPFR = z[ixseps2-1, 0:ixseps_sep2+1,1]

        # Append the first point after the core to close the line
        r_EPFR = np.append(r_EPFR, r[ixseps2-1, 5*ixseps_sep2+1:ixseps_sep+2,1])
        z_EPFR = np.append(z_EPFR, z[ixseps2-1, 5*ixseps_sep2+1:ixseps_sep+2,1])


        #SF separatrix lines have "jumps"
        #+2 comes from the ghost cells
        ax.plot(r[ixseps1-1, 0:ixseps_sep+2,1], z[ixseps1-1, 0:ixseps_sep+2,1], color="magenta", label="ixseps1",linewidth=8.0) #SOL -> ixseps_upper = ixseps1
        ax.plot(r[ixseps1-1, ixseps_sep+2:-1,1], z[ixseps1-1, ixseps_sep+2:-1,1], color="magenta", label="ixseps1",linewidth=8.0) #SOL -> ixseps_upper = ixseps1

        #ixseps2 = ixseps_lower in this case
        ax.plot(r_EPFR, z_EPFR, color="darkorchid", label="ixseps2",linewidth=8.0) #East PFR section of ixseps2
        ax.plot(r_core, z_core, color="darkorchid",linewidth=8.0) #Core part of ixseps2
        ax.plot(r[ixseps2-1, 5*ixseps_sep2+1:ixseps_sep+2,1], z[ixseps2-1, 5*ixseps_sep2+1:ixseps_sep+2,1], color="darkorchid",linewidth=8.0) #West PFR section of ixseps2
        ax.plot(r[ixseps2-1, ixseps_sep+2:-1,1], z[ixseps2-1, ixseps_sep+2:-1,1], color="darkorchid",linewidth=8.0) #South PFR section of ixseps2

        #Target plates in y = 0 and y = ny
        ax.plot(r[:, 0,3], z[:, 0,3], color="c",linewidth=10.0)
        ax.plot(r[:, ny+3,3], z[:, ny+3,3], color="c",linewidth=10.0)

        #Branch cuts in y direction
        ax.plot(r[0:int(nx/3*2)-1, jyseps1_1+1,4], z[0:int(nx/3*2)-1, jyseps1_1+1,4], color="gold", label="jyseps1_1",linewidth=8.0)
        ax.plot(r[0:int(nx/3), jyseps1_2+1,4], z[0:int(nx/3), jyseps1_2+1,4], color="b", label="jyseps1_2",linewidth=8.0)
        ax.plot(r[0:int(nx/3*2)-2, jyseps2_1+1,4], z[0:int(nx/3*2)-2, jyseps2_1+1,4], color="greenyellow", label="jyseps2_1",linewidth=8.0)
        ax.plot(r[0:int(nx/3), jyseps2_2+3,4], z[0:int(nx/3), jyseps2_2+3,4], color="r", label="jyseps2_2",linewidth=8.0)

        #Target plates in y = ny_inner and y = ny_inner + 1
        ax.plot(r[:, ny_inner+1,3], z[:, ny_inner+1,3], color="c", label="ny_inner",linewidth=10.0)
        ax.plot(r[:, ny_inner+3,3], z[:, ny_inner+3,3], color="c",linewidth=10.0)

        #plt.plot(R,Z)
        # If necessary, transpose for plotting (depends on your BOUT setup)
        if R.shape[0] < R.shape[1]:  # Often (x, y), but plotting usually expects (y, x)
            R = R.T
            Z = Z.T


        ax.set_title("BOUT++ Grid from " + str(grid_file))
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect('equal')
        plt.legend()
        #plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_points(ds,grid_file):
    # Extract R and Z coordinates
    R = ds["Rxy"].values
    Z = ds["Zxy"].values
    ixseps1 = ds["ixseps1"].values
    ixseps2 = ds["ixseps2"].values
    jyseps1_1 = ds["jyseps1_1"].values
    jyseps1_2 = ds["jyseps1_2"].values
    ny_inner = ds["ny_inner"].values
    jyseps2_1 = ds["jyseps2_1"].values
    jyseps2_2 = ds["jyseps2_2"].values

    #xbout.plotting.plotfuncs.plot2d_polygon(ds["Rxy"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(R[ixseps1, :], Z[ixseps1, :], color="magenta", label="ixseps1")
    ax.plot(R[ixseps2, :], Z[ixseps2, :], color="yellow", label="ixseps2")
    ax.plot(R[:, jyseps1_1], Z[:, jyseps1_1], color="k", label="jyseps1_1")
    ax.plot(R[:, jyseps1_2], Z[:, jyseps1_2], color="b", label="jyseps1_2")
    ax.plot(R[:, ny_inner], Z[:, ny_inner], color="c", label="ny_inner")
    ax.plot(R[:, jyseps2_1], Z[:, jyseps2_1], color="g", label="jyseps2_1")
    ax.plot(R[:, jyseps2_2], Z[:, jyseps2_2], color="r", label="jyseps2_2")
    plt.plot(R,Z, "x")
    ax = plt.gca()
    ax.set_title("BOUT++ Grid from " + str(grid_file))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    plt.legend()
    plt.show()


def plot(ds, edgecolor="black", ax: object = None, show=True):
    """
    Plot UEDGE grid from 'dict' obtained from method 'ImportGridue'

    Parameters
    ----------
    GridueParams : dict
        Gridue header and body information as a dictionary.

    edgecolor : str, optional
        Color of grid.

    ax : object, optional
        Matplotlib axes to plot on.

    """
    r = ds["rm"].values
    z = ds["zm"].values
    print(r.shape)
    Nx = len(r)
    Ny = len(r[0])
    patches = []
    plt.figure(figsize=(6, 10))
    if ax is None:
        ax = plt.gca()
    idx = [np.array([1, 2, 4, 3, 1])]
    for i in range(Nx):
        for j in range(Ny):
            p = matplotlib.patches.Polygon(
                np.concatenate((r[i][j][idx], z[i][j][idx])).reshape(2, 5).T,
                fill=False,
                closed=True,
                edgecolor=edgecolor, 
                linewidth=2.0
            )
            ax.add_patch(p)  # draw the contours
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_ylim(z.min(), z.max())
    ax.set_xlim(r.min(), r.max())
    plt.show() if show else None
    return ax

plot_lines(d1,grid_file1,topology="SF")
#plot_points(d1,grid_file1)
plot(d1, edgecolor="black", show=True)





