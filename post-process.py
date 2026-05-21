"""
Post-process the output of the simulation to extract velocity, temperature, and pressure contours, as well as streamlines. This script assumes that the simulation output is stored in a .csv/.txt format with columns for coordinates (x, y, z),velocity components (u, v, w), temperature (T), and pressure (p). All the data fields are expected to have a linear indexing format (i + j * nx + k * nx * ny), which will be rearranged into 3D arrays for visualization.

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

class Plotter:
    def __init__(self, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d):
        self.x_3d = x_3d
        self.y_3d = y_3d
        self.z_3d = z_3d
        self.u_3d = u_3d
        self.v_3d = v_3d
        self.w_3d = w_3d
        self.T_3d = T_3d
        self.p_3d = p_3d

    # plot velocity contours of {'XY', 'XZ', 'YZ'} planes at a given slice
    def plot_velocity_contours(self, plane, slice):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

        if plane == 'XY':
            c0 = axes[0].contourf(self.x_3d[:, :, slice], self.y_3d[:, :, slice], self.u_3d[:, :, slice], levels=50, cmap='jet')
            fig.colorbar(c0, ax=axes[0], label='Velocity u')

            c1 = axes[1].contourf(self.x_3d[:, :, slice], self.y_3d[:, :, slice], self.v_3d[:, :, slice], levels=50, cmap='jet')
            fig.colorbar(c1, ax=axes[1], label='Velocity v')

            c2 = axes[2].contourf(self.x_3d[:, :, slice], self.y_3d[:, :, slice], self.w_3d[:, :, slice], levels=50, cmap='jet')
            fig.colorbar(c2, ax=axes[2], label='Velocity w')

            fig.supxlabel('x')
            fig.supylabel('y')
        
        elif plane == 'XZ':
            c0 = axes[0].contourf(self.x_3d[:, slice, :], self.z_3d[:, slice, :], self.u_3d[:, slice, :], levels=50, cmap='jet')
            fig.colorbar(c0, ax=axes[0], label='Velocity u')

            c1 = axes[1].contourf(self.x_3d[:, slice, :], self.z_3d[:, slice, :], self.v_3d[:, slice, :], levels=50, cmap='jet')
            fig.colorbar(c1, ax=axes[1], label='Velocity v')

            c2 = axes[2].contourf(self.x_3d[:, slice, :], self.z_3d[:, slice, :], self.w_3d[:, slice, :], levels=50, cmap='jet')
            fig.colorbar(c2, ax=axes[2], label='Velocity w')

            fig.supxlabel('x')
            fig.supylabel('z')

        elif plane == 'YZ':
            c0 = axes[0].contourf(self.z_3d[slice, :, :], self.y_3d[slice, :, :], self.u_3d[slice, :, :], levels=50, cmap='jet')
            fig.colorbar(c0, ax=axes[0], label='Velocity u')

            c1 = axes[1].contourf(self.z_3d[slice, :, :], self.y_3d[slice, :, :], self.v_3d[slice, :, :], levels=50, cmap='jet')
            fig.colorbar(c1, ax=axes[1], label='Velocity v')

            c2 = axes[2].contourf(self.z_3d[slice, :, :], self.y_3d[slice, :, :], self.w_3d[slice, :, :], levels=50, cmap='jet')
            fig.colorbar(c2, ax=axes[2], label='Velocity w')

            fig.supxlabel('z')
            fig.supylabel('y')

        fig.suptitle(f'Velocity Contours of {plane} plane at slice {slice}')
        fig.tight_layout()
        plt.show()

    # plot XY plane streamlines at a given z-slice
    def plot_streamlines(self, z_slice):
        plt.figure(figsize=(6, 5))
        plt.streamplot(self.x_3d[:, 0, z_slice], self.y_3d[0, :, z_slice], self.u_3d[:, :, z_slice].T, self.v_3d[:, :, z_slice].T, color='k', density=1.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Streamlines at z-slice {z_slice}')
        plt.tight_layout()
        plt.show()

    # plot XY plane temperature contours at a given z-slice
    def plot_temperature_contours(self, plane, slice):
        plt.figure(figsize=(6, 5))
        if plane == 'XY':
            plt.contourf(self.x_3d[:, :, slice], self.y_3d[:, :, slice], self.T_3d[:, :, slice], levels=50, cmap='inferno')
            plt.xlabel('x')
            plt.ylabel('y')
        elif plane == 'XZ':
            plt.contourf(self.x_3d[:, slice, :], self.z_3d[:, slice, :], self.T_3d[:, slice, :], levels=50, cmap='inferno')
            plt.xlabel('x')
            plt.ylabel('z')
        elif plane == 'YZ':
            plt.contourf(self.z_3d[slice, :, :], self.y_3d[slice, :, :], self.T_3d[slice, :, :], levels=50, cmap='inferno')
            plt.xlabel('z')
            plt.ylabel('y')
        plt.colorbar(label='Temperature T')
        plt.title(f'Temperature Contour of {plane} plane at slice {slice}')
        plt.tight_layout()
        plt.show()

    # plot XY plane pressure contours at a given z-slice
    def plot_pressure_contours(self, z_slice):
        plt.figure(figsize=(6, 5))
        plt.contourf(self.x_3d[:, :, z_slice], self.y_3d[:, :, z_slice], self.p_3d[:, :, z_slice], levels=50, cmap='viridis')
        plt.colorbar(label='Pressure p')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Pressure Contour at z-slice {z_slice}')
        plt.tight_layout()
        plt.show()

# example usage
filename = 'lbm_results.csv' # replace with actual filename

nx, ny, nz, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d = load_simulation_data(filename)
plotter = Plotter(x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d)

 # for XY plane use slice 0 to nz-1, for XZ plane use slice 0 to ny-1, for YZ plane use slice 0 to nx-1
plotter.plot_velocity_contours(plane='YZ', slice=nx//2) # plot velocity contours
plotter.plot_streamlines(z_slice=nz//2) # plot streamlines
plotter.plot_temperature_contours(plane='YZ', slice=nx//2) # plot temperature contours
plotter.plot_pressure_contours(z_slice=nz//2) # plot pressure contours
