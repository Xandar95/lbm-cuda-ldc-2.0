/* GPU accelerated Heated lid driven cavity flow simulation using Lattice Boltzmann Method (LBM) */

#include <iostream>
#include <cmath>
#include <fstream>
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
    float alpha = 0.1f; // thermal diffusivity in lattice units
    float beta = 1.0e-4f; // thermal expansion coefficient in lattice units
    float gravity = 1.0e-3f; // gravitational acceleration in lattice units
    float T_ref = 0.0f; // reference temperature in lattice units
    float kappa_wall = 1.0f; // thermal conductivity of the wall in lattice units
    float kappa_top = 0.1f; // thermal conductivity of the top lid in lattice units
    float q_wall = 1.0e-4f; // heat flux at the wall in lattice units
    float q_top = -1.0e-5f; // heat flux at the top lid in lattice units

    // relaxation parameters
    float omega_f = 1.0f / (3.0f * nu + 0.5f); // relaxation parameter for momentum equation
    float omega_g = 1.0f / (3.0f * alpha + 0.5f); // relaxation parameter for energy equation

    // Lattice positions to be set in the python post-processing script
    
    // Set field variables and distribution functions
    int N = nx * ny * nz; // total number of grid points
    float* rho = new float[N]; // density
    float* T = new float[N]; // temperature
    float* u = new float[N]; // velocity in x direction
    float* v = new float[N]; // velocity in y direction
    float* w = new float[N]; // velocity in z direction

    float* f = new float[N * 27]; // distribution function for momentum
    float* g = new float[N * 27]; // distribution function for temperature
    
    // Lambda function to compute the 1D index from 3D indices
    auto idx = [nx, ny](int i, int j, int k) {
        return i + j * nx + k * nx * ny;
    };
    // Lambda function to compute the 1D index for distribution functions
    auto pdf = [N](int l, int idx) {
        return l * N + idx;
    };

    // Initialize the fields and distribution functions
    // rho = 1.0, T = 1.0, u = v = w = 0.0 everywhere except the top lid where u = u_lid
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) { 
            for (int i = 0; i < nx; i++) {
                rho[idx(i, j, k)] = 1.0f;
                T[idx(i, j, k)] = 1.0f;
                u[idx(i, j, k)] = (j == ny - 1) ? u_lid : 0.0f;
                v[idx(i, j, k)] = 0.0f;
                w[idx(i, j, k)] = 0.0f;

                float u2 = u[idx(i, j, k)] * u[idx(i, j, k)] + v[idx(i, j, k)] * v[idx(i, j, k)] + w[idx(i, j, k)] * w[idx(i, j, k)]; 
                for (int l = 0; l < 27; l++) {
                    float cu = cx[l] * u[idx(i, j, k)] + cy[l] * v[idx(i, j, k)] + cz[l] * w[idx(i, j, k)];
                    f[pdf(l, idx(i, j, k))] = weights[l] * rho[idx(i, j, k)] * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u2); // second order equilibrium distribution for momentum
                    g[pdf(l, idx(i, j, k))] = weights[l] * T[idx(i, j, k)] * (1.0f + 3.0f * cu); // first order equilibrium distribution for temperature
                }
            }
        }
    }

    // Initialize the GPU
    printf("Initializing GPU...\n");
    lbm_init_gpu(nx, ny, nz, cx, cy, cz, opp, weights);

    // Copy initial distribution functions to GPU
    printf("Copying initial data to GPU...\n");
    lbm_copy_host_to_device(f, g, nx, ny, nz);

    // Begin time-stepping loop
    printf("Starting time-stepping loop on GPU...\n");

    for (int t = 0; t < nsteps; t++) {
        lbm_run_step_gpu(nx, ny, nz, omega_f, omega_g, u_lid, beta, gravity, T_ref, q_wall, q_top, kappa_wall, kappa_top);

        // Check residual and progress output every 1000 steps
        if (t % 1000 == 0) {
            float residual_u = lbm_compute_u_residual_gpu(nx, ny, nz);
            float residual_T = lbm_compute_T_residual_gpu(nx, ny, nz); 
            printf("Time step: %d, u residual: %e, T residual: %e\n", t, residual_u, residual_T);

            if (residual_u < 1.0e-4f && residual_T < 1.0e-5f) {
                printf("Simulation converged at time step %d\n", t);
                break;
            }
        }
    }

    // Copy final results back to the host
    printf("Copying results back to the host...\n");
    lbm_copy_device_to_host(f, g, nx, ny, nz);

    // Free device memory 
    lbm_free_gpu();

    // Output velocity, temperature and pressure fields
    std::ofstream output_file;
    output_file.open("lbm_results.csv");
    // Compute macroscopic variables from distribution functions and write to file
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                float rho_local = 0.0f;
                float T_local = 0.0f;
                float u_local = 0.0f;
                float v_local = 0.0f;
                float w_local = 0.0f;

                for (int l = 0; l < 27; l++) {
                    rho_local += f[pdf(l, idx(i, j, k))];
                    T_local += g[pdf(l, idx(i, j, k))];
                    u_local += f[pdf(l, idx(i, j, k))] * cx[l];
                    v_local += f[pdf(l, idx(i, j, k))] * cy[l];
                    w_local += f[pdf(l, idx(i, j, k))] * cz[l];
                }
                u_local /= rho_local;
                v_local /= rho_local;
                w_local /= rho_local;
                float p_local = cs * cs * rho_local; // pressure from equation of state

                output_file << u_local << "," << v_local << "," << w_local << "," << T_local << "," << p_local << "\n";
            }
        }
    }
    output_file.close();

    return 0;
}