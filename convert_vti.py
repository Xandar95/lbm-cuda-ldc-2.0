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

    grid['Magnitude'] = np.sqrt(u.flatten(order='F')**2 + v.flatten(order='F')**2 + w.flatten(order='F')**2)

    # save the grid to a .vti file
    grid.save(filename.replace('.csv', '.vti'))

def main():
    output_dir = 'vti_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)

    files = sorted([f for f in os.listdir('..') if f.endswith('.csv')])
    for filename in files:
        print(f"Processing {filename}...")
        nx, ny, nz, x, y, z, u, v, w, T, p = load_simulation_data(os.path.join('..', filename))
        save_to_vti(filename, nx, ny, nz, x, y, z, u, v, w, T, p)
    print("Conversion complete. .vti files saved in 'vti_output' directory.")

if __name__ == "__main__":
    main()
