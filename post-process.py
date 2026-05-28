"""
Post-process the output of the simulation to extract velocity, temperature, and pressure contours, as well as streamlines. This script assumes that the simulation output is stored in a .csv/.txt format with columns for coordinates (x, y, z),velocity components (u, v, w), temperature (T), and pressure (p). All the data fields are expected to have a linear indexing format (i + j * nx + k * nx * ny), which will be rearranged into 3D arrays for visualization.

author: Sandun
"""
import numpy as np
import matplotlib.pyplot as plt
import os # for file handling
import matplotlib.animation as animation 

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

    # animate velocity contours of {'XY', 'XZ', 'YZ'} planes at a given slice
    def animate_velocity_contours(self, files, plane, slice):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        colorbars = []

        def draw_frame(frame):
            filename = files[frame]
            nx, ny, nz, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d = load_simulation_data(filename)

            # Remove colorbar axes from the previous frame before drawing new ones
            for cbar in colorbars:
                cbar.remove()
            colorbars.clear()

            for ax in axes:
                ax.clear()
            if plane == 'XY':
                c0 = axes[0].contourf(x_3d[:, :, slice], y_3d[:, :, slice], u_3d[:, :, slice], levels=24, cmap='jet')
                colorbars.append(fig.colorbar(c0, ax=axes[0], label='Velocity u'))

                c1 = axes[1].contourf(x_3d[:, :, slice], y_3d[:, :, slice], v_3d[:, :, slice], levels=24, cmap='jet')
                colorbars.append(fig.colorbar(c1, ax=axes[1], label='Velocity v'))

                c2 = axes[2].contourf(x_3d[:, :, slice], y_3d[:, :, slice], w_3d[:, :, slice], levels=24, cmap='jet')
                colorbars.append(fig.colorbar(c2, ax=axes[2], label='Velocity w'))

                fig.supxlabel('x')
                fig.supylabel('y')

            elif plane == 'XZ':
                c0 = axes[0].contourf(x_3d[:, slice, :], z_3d[:, slice, :], u_3d[:, slice, :], levels=24, cmap='jet')
                colorbars.append(fig.colorbar(c0, ax=axes[0], label='Velocity u'))

                c1 = axes[1].contourf(x_3d[:, slice, :], z_3d[:, slice, :], v_3d[:, slice, :], levels=24, cmap='jet')
                colorbars.append(fig.colorbar(c1, ax=axes[1], label='Velocity v'))

                c2 = axes[2].contourf(x_3d[:, slice, :], z_3d[:, slice, :], w_3d[:, slice, :], levels=24, cmap='jet')
                colorbars.append(fig.colorbar(c2, ax=axes[2], label='Velocity w'))

                fig.supxlabel('x')
                fig.supylabel('z')

            elif plane == 'YZ':
                c0 = axes[0].contourf(z_3d[slice, :, :], y_3d[slice, :, :], u_3d[slice, :, :], levels=24, cmap='jet')
                colorbars.append(fig.colorbar(c0, ax=axes[0], label='Velocity u'))

                c1 = axes[1].contourf(z_3d[slice, :, :], y_3d[slice, :, :], v_3d[slice, :, :], levels=24, cmap='jet')
                colorbars.append(fig.colorbar(c1, ax=axes[1], label='Velocity v'))

                c2 = axes[2].contourf(z_3d[slice, :, :], y_3d[slice, :, :], w_3d[slice, :, :], levels=24, cmap='jet')
                colorbars.append(fig.colorbar(c2, ax=axes[2], label='Velocity w'))

                fig.supxlabel('z')
                fig.supylabel('y')

            else:
                raise ValueError("Invalid plane. Choose from 'XY', 'XZ', 'YZ'.")

            fig.suptitle(f'Velocity Contours of {plane} plane at slice {slice} \n Frame: {frame+1}/{len(files)}')
            fig.tight_layout()
            return [c0, c1, c2]

        anim_velocity = animation.FuncAnimation(fig, draw_frame, frames=range(len(files)), interval=50,
                                                blit=False, repeat=False, cache_frame_data=False)
        return anim_velocity

    # animate XY plane streamlines at a given z-slice
    def animate_streamlines(self, files, plane, slice):
        fig = plt.figure(figsize=(6, 5))

        def draw_frame(frame):
            filename = files[frame]
            nx, ny, nz, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d = load_simulation_data(filename)

            plt.clf()
            if plane == 'XY':
                plt.streamplot(x_3d[:, 0, slice], y_3d[0, :, slice], u_3d[:, :, slice].T, v_3d[:, :, slice].T, color='k', density=1.5)
                plt.xlabel('x')
                plt.ylabel('y')
            elif plane == 'XZ':
                plt.streamplot(x_3d[:, slice, 0], z_3d[0, slice, :], u_3d[:, slice, :].T, w_3d[:, slice, :].T, color='k', density=1.5)
                plt.xlabel('x')
                plt.ylabel('z')
            elif plane == 'YZ':
                plt.streamplot(z_3d[slice, 0, :], y_3d[slice, :, 0], w_3d[slice, :, :].T, v_3d[slice, :, :].T, color='k', density=1.5)
                plt.xlabel('z')
                plt.ylabel('y')
            else:
                raise ValueError("Invalid plane. Choose from 'XY', 'XZ', 'YZ'.")

            plt.title(f'Streamlines of {plane} plane at slice {slice} \n Frame: {frame+1}/{len(files)}')
            plt.tight_layout()
            return plt.gca()

        anim_streamlines = animation.FuncAnimation(fig, draw_frame, frames=range(len(files)), interval=50,
                                                    blit=False, repeat=False, cache_frame_data=False)
        return anim_streamlines

    # animate XY plane temperature contours at a given z-slice
    def animate_temperature_contours(self, files, plane, slice):
        fig = plt.figure(figsize=(6, 5))
        colorbars = []

        def draw_frame(frame):
            filename = files[frame]
            nx, ny, nz, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d = load_simulation_data(filename)

            # Remove colorbar axes from the previous frame before drawing new ones
            for cbar in colorbars:
                cbar.remove()
            colorbars.clear()
        
            plt.clf()
            if plane == 'XY':
                plt.contourf(self.x_3d[:, :, slice], self.y_3d[:, :, slice], self.T_3d[:, :, slice], levels=24, cmap='inferno')
                plt.xlabel('x')
                plt.ylabel('y')
            elif plane == 'XZ':
                plt.contourf(self.x_3d[:, slice, :], self.z_3d[:, slice, :], self.T_3d[:, slice, :], levels=24, cmap='inferno')
                plt.xlabel('x')
                plt.ylabel('z')
            elif plane == 'YZ':
                plt.contourf(self.z_3d[slice, :, :], self.y_3d[slice, :, :], self.T_3d[slice, :, :], levels=24, cmap='inferno')
                plt.xlabel('z')
                plt.ylabel('y')
            else:
                raise ValueError("Invalid plane. Choose from 'XY', 'XZ', 'YZ'.")
            colorbars.append(plt.colorbar(label='Temperature T'))
            plt.title(f'Temperature Contour of {plane} plane at slice {slice} \n Frame: {frame+1}/{len(files)}')
            plt.tight_layout()
            return plt.gca().collections
            
        anim_temperature = animation.FuncAnimation(fig, draw_frame, frames=range(len(files)), interval=50,
                                                    blit=False, repeat=False, cache_frame_data=False)
        return anim_temperature

    # animate XY plane pressure contours at a given z-slice
    def animate_pressure_contours(self, files, plane, slice):
        fig = plt.figure(figsize=(6, 5))
        colorbars = []

        def draw_frame(frame):
            filename = files[frame]
            nx, ny, nz, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d = load_simulation_data(filename)

            # Remove colorbar axes from the previous frame before drawing new ones
            for cbar in colorbars:
                cbar.remove()
            colorbars.clear()
        
            plt.clf()
            if plane == 'XY':
                plt.contourf(self.x_3d[:, :, slice], self.y_3d[:, :, slice], self.p_3d[:, :, slice], levels=24, cmap='viridis')
                plt.xlabel('x')
                plt.ylabel('y')
            elif plane == 'XZ':
                plt.contourf(self.x_3d[:, slice, :], self.z_3d[:, slice, :], self.p_3d[:, slice, :], levels=24, cmap='viridis')
                plt.xlabel('x')
                plt.ylabel('z')
            elif plane == 'YZ':
                plt.contourf(self.z_3d[slice, :, :], self.y_3d[slice, :, :], self.p_3d[slice, :, :], levels=24, cmap='viridis')
                plt.xlabel('z')
                plt.ylabel('y')
            else:
                raise ValueError("Invalid plane. Choose from 'XY', 'XZ', 'YZ'.")
            colorbars.append(plt.colorbar(label='Pressure p'))
            plt.title(f'Pressure Contour of {plane} plane at slice {slice} \n Frame: {frame+1}/{len(files)}')
            plt.tight_layout()
            return plt.gca()
        
        anim_pressure = animation.FuncAnimation(fig, draw_frame, frames=range(len(files)), interval=50,
                                                    blit=False, repeat=False, cache_frame_data=False)
        return anim_pressure

# main function to load data and create visualizations
def main():
    # get list of .csv files in the sim_output directory
    output_dir = 'sim_output'
    files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv')])
    if not files:
        raise FileNotFoundError('No .csv files found in the sim_output directory.')

    # load the first file to get grid dimensions and create plotter instance
    nx, ny, nz, x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d = load_simulation_data(files[0])
    plotter = Plotter(x_3d, y_3d, z_3d, u_3d, v_3d, w_3d, T_3d, p_3d)

    # create animations for velocity contours, streamlines, temperature contours, and pressure contours
    #anim_velocity = plotter.animate_velocity_contours(files, plane='XY', slice=nz//2)
    #anim_streamlines = plotter.animate_streamlines(files, plane='XY', slice=nz//2)
    anim_temperature = plotter.animate_temperature_contours(files, plane='XY', slice=nz//2)
    #anim_pressure = plotter.animate_pressure_contours(files, plane='XY', slice=nz//2)
    plt.show()

if __name__ == "__main__":
    main()

        
    







