"""
Script to convert .csv files to .vti format for use in Paraview.

author: Sandun
"""
import numpy as np
import os
import pyvista as pv

# function to load simulation data from .csv file and rearrange the linear data into 3D arrays
def load_simulation_data(filename):
    data = np.loadtxt(filename, delimiter=',') # no header, columns: x, y, z, u, v, w, T, p

    # determine grid dimensions
    nx = len(np.unique(data[:, 0]))
    ny = len(np.unique(data[:, 1]))
    nz = len(np.unique(data[:, 2]))

    # reshape data into 3D arrays (Fortran order to match linear indexing)
    x = data[:, 0].reshape((nx, ny, nz), order='F')
    y = data[:, 1].reshape((nx, ny, nz), order='F')
    z = data[:, 2].reshape((nx, ny, nz), order='F')
    u = data[:, 3].reshape((nx, ny, nz), order='F')
    v = data[:, 4].reshape((nx, ny, nz), order='F')
    w = data[:, 5].reshape((nx, ny, nz), order='F')
    T = data[:, 6].reshape((nx, ny, nz), order='F')
    p = data[:, 7].reshape((nx, ny, nz), order='F')

    return nx, ny, nz, x, y, z, u, v, w, T, p

# function to create a structured grid and save it as a .vti file
def save_to_vti(filename, nx, ny, nz, x, y, z, u, v, w, T, p):
    # create a VTK image data
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.origin = (x.min(), y.min(), z.min())
    # compute grid spacing
    dx, dy, dz = (x[1, 0, 0] - x[0, 0, 0], y[0, 1, 0] - y[0, 0, 0], z[0, 0, 1] - z[0, 0, 0])
    grid.spacing = (dx, dy, dz)

    # add field data to the grid
    grid['u'] = u.flatten(order='F') 
    grid['v'] = v.flatten(order='F')
    grid['w'] = w.flatten(order='F')
    grid['T'] = T.flatten(order='F')
    grid['p'] = p.flatten(order='F')

    grid['Velocity'] = np.column_stack((u.flatten(order='F'), v.flatten(order='F'), w.flatten(order='F'))) # velocity vector

    # save the grid
    grid.save(filename)

def main():
    output_dir = 'vti_output'
    os.makedirs(output_dir, exist_ok=True) # create output directory if it doesn't exist

    # get list of .csv files in the sim_output directory
    input_dir = 'sim_output'
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')])
    if not files:
        raise FileNotFoundError(f"No .csv files found in {input_dir}")

    for filename in files:
        print(f"Processing {filename}...")
        nx, ny, nz, x, y, z, u, v, w, T, p = load_simulation_data(filename)
        output_filename = os.path.join(output_dir, os.path.basename(filename).replace('.csv', '.vti'))
        save_to_vti(output_filename, nx, ny, nz, x, y, z, u, v, w, T, p)
    print("Conversion complete. .vti files saved in 'vti_output' directory.")

if __name__ == "__main__":
    main()
