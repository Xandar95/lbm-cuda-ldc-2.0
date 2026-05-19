"""
Post-process the output of the simulation to extract velocity, temperature, and pressure contours, as well as streamlines. This script assumes that the simulation output is stored in a .csv format with columns for coordinates (x, y, z),velocity components (u, v, w), temperature (T), and pressure (p). All the data fields are expected to have a linear indexing format (i + j * nx + k * nx * ny), which will be rearranged into 3D arrays for visualization.

author: Sandun
"""
import numpy as np
import matplotlib.pyplot as plt

# function to load simulation data from .csv file and rearrange the linear data into 3D arrays
def load_simulation_data(filename):
    data = np.loadtxt(filename, delimiter=',') # no header, columns: x, y, z, u, v, w, T, p
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    u = data[:, 3]
    v = data[:, 4]
    w = data[:, 5]
    T = data[:, 6]
    p = data[:, 7]

    # determine grid dimensions
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    nz = len(np.unique(z))

    # reshape data into 3D arrays (Fortran order to match linear indexing)
    x_3d = x.reshape((nx, ny, nz), order='F')
    y_3d = y.reshape((nx, ny, nz), order='F')
    z_3d = z.reshape((nx, ny, nz), order='F')
    u_3d = u.reshape((nx, ny, nz), order='F')
    v_3d = v.reshape((nx, ny, nz), order='F')
    w_3d = w.reshape((nx, ny, nz), order='F')
    T_3d = T.reshape((nx, ny, nz), order='F')
    p_3d = p.reshape((nx, ny, nz), order='F')

    return nx, ny, nz, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d

# function to plot velocity contours at a given z-slice
def plot_velocity_contours(x_3d, y_3d, u_3d, v_3d, z_slice):
    plt.figure()
    plt.contourf(x_3d[:, :, z_slice], y_3d[:, :, z_slice], u_3d[:, :, z_slice], levels=50, cmap='jet')
    plt.colorbar(label='Velocity u')
    plt.title(f'Velocity Contours at z={z_slice}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# example usage
filename = 'lbm_results.csv' # replace with actual filename

nx, ny, nz, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d = load_simulation_data(filename)
plot_velocity_contours(x_3d, y_3d, u_3d, v_3d, z_slice=nz//2) # plot velocity contours at the first z-slice
