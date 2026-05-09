/* GPU accelerated Heated lid driven cavity flow simulation using Lattice Boltzmann Method (LBM) */

#include <iostream>
#include <cmath>
#include "lbm_kernel.cuh"

int main() {
    // spatial and temporal resolution
    int nx = 126; int ny = 126; int nz = 126; // grid size
    int nsteps = 1.0e6; // number of time steps

    // domain parameters
    float Lx = 1.0f; float Ly = 1.0f; float Lz = 1.0f; // domain size
    float dx = Lx / (nx - 1); float dy = Ly / (ny - 1); float dz = Lz / (nz - 1); // grid spacing

    // lattice parameters
    float cs = 1.0f /sqrt(3.0f); // lattice speed of sound
    float weights[27] = {8.0f/27.0f, // w0
                        2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, // w1-w6
                        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, // w7-w18
                        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
                        1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f}; // w19-w26
    
    int cx[27] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1}; // lattice velocity in x direction
    int cy[27] = {0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1}; // lattice velocity in y direction
    int cz[27] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 0, 0, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1}; // lattice velocity in z direction

    int opp[27] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25}; // opposite lattice directions

    // physical parameters 
    float u_lid = 0.01f; // lid velocity in lattice units
    float Re = 100.0f; // Reynolds number
    float nu = u_lid * (ny - 1) / Re; // kinematic viscosity in lattice units




    return 0;
}