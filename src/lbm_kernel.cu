// LBM Kernel Implementation in CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath> 
#include "lbm_kernel.cuh"

// D3Q27 model parameters stored in constant memory
// D3Q27 discrete velocities 
__constant__ int d_cx[27];
__constant__ int d_cy[27];
__constant__ int d_cz[27];
__constant__ int d_opp[27]; // Opposite directions
__constant__ float d_weights[27]; // D3Q27 weights

// Device memory allocation for distribution functions f (momentum) and g (temperature)
static float *d_f_in = nullptr;
static float *d_f_out = nullptr;
static float *d_g_in = nullptr; 
static float *d_g_out = nullptr;
// Device memory for velocity residual calculation
static float *d_u_diff_sq = nullptr;  
static float *d_u_mag_sq = nullptr; 
// Device memory for tempurature residual calculation
static float *d_T_diff_sq = nullptr;
static float *d_T_mag_sq = nullptr;

// ----- Bulk kernel for LBM step (Pull scheme + SOA) -----
__global__ void lbm_kernel_soa(const float* __restrict__ f_in,
                               float* __restrict__ f_out,
                               const float* __restrict__ g_in,
                               float* __restrict__ g_out,
                               int nx, int ny, int nz,
                               float omega_f, float omega_g,
                               float beta, float gravity, float T_ref)
{
    int N = nx * ny * nz; // total number of lattice points

    // -------- 3D grid and block configuration --------
    // define 3D global indices 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return; // boundary check for thread space

    int curr_idx = i + j * nx + k * nx * ny; // current index

    // -------- 2D grid and block configuration --------
    // define 2D global indices (i, j) 
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    // if (i >= nx || j >= ny) return; // boundary check for thread space

    // loop over k dimension
    // for (int k = 0; k < nz; k++) {
    //      int curr_idx = i + j * nx + k * nx * ny;} // wrap the k loop till the end of the kernel

    // -------- 1D grid and block configuration --------
    // define global 1D linear index
    // int curr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check for the 1D thread space 
    // if (curr_idx >= N) return;

    // deconstruct 1D index back into 3D coordinates (i, j, k)
    // Formula: i = idx % nx,  j = (idx / nx) % ny,  k = idx / (nx * ny)
    // int i = curr_idx % nx;
    // int j = (curr_idx / nx) % ny;
    // int k = curr_idx / (nx * ny);

    // register variables for local calculations
    float f_streamed[27];
    float g_streamed[27];
    float rho = 0.0; // density
    float T = 0.0; // temperature
    float u = 0.0, v = 0.0, w = 0.0; 

    // Pull streaming & moment calculation
    #pragma unroll
    for (int l = 0; l < 27; l++){
        int ip = i - d_cx[l];
        int jp = j - d_cy[l];
        int kp = k - d_cz[l];

        // apply periodic BC for simplicity
        // Actual BC will overwrite periodicity later
        if (ip >= 0 && ip < nx && jp >= 0 && jp < ny && kp >= 0 && kp < nz) {
            int neigh_idx = ip + jp * nx + kp * nx * ny;
            f_streamed[l] = f_in[l * N + neigh_idx];
            g_streamed[l] = g_in[l * N + neigh_idx];
        } else {
            f_streamed[l] = f_in[l * N + curr_idx]; // boundary fallback
            g_streamed[l] = g_in[l * N + curr_idx]; // boundary fallback
        }

        // accumulate moments
        rho += f_streamed[l];
        T += g_streamed[l];
        u += f_streamed[l] * d_cx[l];
        v += f_streamed[l] * d_cy[l];
        w += f_streamed[l] * d_cz[l];
        }

        if (rho > 1e-9) {u /= rho; v /= rho; w /= rho;} // avoid division by zero
        else {u = 0.0; v = 0.0; w = 0.0;}

        // Forcing term for buoyancy (only in y-direction)
        float Fy = -beta * gravity * (T - T_ref); // buoyancy force based on local temperature difference (Boussinesq approximation)

        // shift the velocity components to accurately recover the NSE with the force term (u = 1 / rho * sum(f_i * c_i) + 0.5 * (F * dt) / rho)
        v += 0.5 * Fy / rho; // Fx = 0, Fz = 0, so only adjust v for buoyancy force

        float u2 = u * u + v * v + w * w;

    // Collision step
    #pragma unroll
    for (int l = 0; l < 27; l++){
        float cu = d_cx[l] * u + d_cy[l] * v + d_cz[l] * w;
        float feq = d_weights[l] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        float geq = d_weights[l] * T * (1.0 + 3.0 * cu); // first-order equilibrium for temperature
        float S = d_weights[l] * (3.0 * (d_cy[l] - v) + 9.0 * cu * d_cy[l]) * Fy; // Guo's forcing term for buoyancy (Fx = 0, Fz = 0)
        // write post collision distribution
        f_out[l * N + curr_idx] = (1.0 - omega_f) * f_streamed[l] + omega_f * feq + (1.0 - 0.5 * omega_f) * S; // include forcing term
        g_out[l * N + curr_idx] = (1.0 - omega_g) * g_streamed[l] + omega_g * geq;
        
    }
}

// ----- Boundary Condition Kernel -----
__global__ void apply_bc_kernel(float* __restrict__ f,
                                float* __restrict__ g,
                                int nx, int ny, int nz,
                                float u_lid,
                                float q_wall, float T_wall, float kappa) // for heat flux for temperature BC
{
    int N = nx * ny * nz; // total number of lattice points
    
    // -------- 3D grid and block configuration --------
    // define 3D global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return; // boundary check for thread space

    int idx = i + j * nx + k * nx * ny; // current index
    // indices for Neumann BC (for temperature gradient)
    int idx_west = idx + 1; // west neighbor (i+1)
    int idx_east = idx - 1; // east neighbor (i-1)
    int idx_top = idx - nx; // top neighbor (j-1)

    // -------- 2D grid and block configuration --------
    // define 2D global indices (i, j) 
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    // if (i >= nx || j >= ny) return; // boundary check for thread space

    // loop over k dimension
    // for (int k = 0; k < nz; k++) {
    //      int idx = i + j * nx + k * nx * ny;} // wrap the k loop till the end of the kernel

    // -------- 1D grid and block configuration --------
    // define global 1D linear index
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check for the 1D thread space
    // if (idx >= N) return;

    // deconstruct 1D index back into 3D coordinates (i, j, k)
    // Formula: i = idx % nx,  j = (idx / nx) % ny,  k = idx / (nx * ny)
    // int i = idx % nx;
    // int j = (idx / nx) % ny;
    // int k = idx / (nx * ny);

    // No slip walls (bounce-back) and constant heat flux at west, east, and top walls
    // West wall (i=0)
    if (i == 0) {

        // compute interior temperature
        float T_west_in = 0.0;
        for (int l = 0; l < 27; l++) {
            T_west_in += g[l * N + idx_west];
        }

        // impose heat flux boundary condition for temperature (q = -k * dT/dx)
        float T_west = T_west_in + q_wall / kappa; // dx = 1

        // No slip bounce-back for velocity and transform Neumann BC into equivalent Dirichlet BC for temperature
        for (int l = 0; l < 27; l++) {
            if (d_cx[l] == 1) {
                f[l * N + idx] = f[d_opp[l] * N + idx]; // No slip bounce-back for velocity

                g[l * N + idx] = -g[d_opp[l] * N + idx] + 2.0 * d_weights[l] * T_west; // anti-bounce back for temperature to impose Neumann BC
            }
        }
    }

    // East wall (i=nx-1)
    if (i == nx -1) {

        // compute interior temperature
        float T_east_in = 0.0;
        for (int l = 0; l < 27; l++) {
            T_east_in += g[l * N + idx_east];
        }

        // impose heat flux boundary condition for temperature (q = -k * dT/dx)
        float T_east = T_east_in + q_wall / kappa; // dx = 1

        // No slip bounce-back for velocity and transform Neumann BC into equivalent Dirichlet BC for temperature
        for (int l = 0; l < 27; l++) {
            if (d_cx[l] == -1) {
                f[l * N + idx] = f[d_opp[l] * N + idx]; // No slip bounce-back for velocity

                g[l * N + idx] = -g[d_opp[l] * N + idx] + 2.0 * d_weights[l] * T_east; // anti-bounce back for temperature to impose Neumann BC
            }
        }
    }

    // South wall (k=0)
    if (k == 0) {
        for (int l = 0; l < 27; l++) {
            if (d_cz[l] == 1)
                f[l * N + idx] = f[d_opp[l] * N + idx]; // No slip bounce-back for velocity

                g[l * N + idx] = -g[d_opp[l] * N + idx] + 2.0 * d_weights[l] * T_wall; // anti-bounce back for temperature to impose Dirichlet BC
        }
    }

    // North wall (k=nz-1)
    if (k == nz - 1) {
        for (int l = 0; l < 27; l++) {
            if (d_cz[l] == -1)
                f[l * N + idx] = f[d_opp[l] * N + idx]; // No slip bounce-back for velocity

                g[l * N + idx] = -g[d_opp[l] * N + idx] + 2.0 * d_weights[l] * T_wall; // anti-bounce back for temperature to impose Dirichlet BC 
        }
    }

    // Bottom wall (j=0)
    if (j == 0) {
        for (int l = 0; l < 27; l++) {
            if (d_cy[l] == 1)
                f[l * N + idx] = f[d_opp[l] * N + idx]; // No slip bounce-back for velocity

                g[l * N + idx] = -g[d_opp[l] * N + idx] + 2.0 * d_weights[l] * T_wall; // anti-bounce back for temperature to impose Dirichlet BC 
        }
    }

    // Top wall (j=ny-1) 
    if (j == ny - 1) {
        // Exclude corners/edges for the moving part of the lid
        if (i > 0 && i < nx - 1 && k > 0 && k < nz - 1) { 
            
            // (use Zou-He: rho_wall = sum(cy==0) + 2*sum(cy==1))
            float rho_wall = 0.0;
            for (int l = 0; l < 27; l++) {
                if (d_cy[l] == 0) rho_wall += f[l * N + idx];
                else if (d_cy[l] == 1) rho_wall += 2.0 * f[l * N + idx];
            }
            
            for (int l = 0; l < 27; l++) {
                if (d_cy[l] == -1) 
                    f[l * N + idx] = f[d_opp[l] * N + idx] + 6.0 * d_weights[l] * rho_wall * d_cx[l] * u_lid; // Since v = 0, w = 0
            }
        } 
        else {
            // Apply stationary bounce-back to the top edges to prevent leakage
            for (int l = 0; l < 27; l++) {
                if (d_cy[l] == -1) 
                    f[l * N + idx] = f[d_opp[l] * N + idx];
            }
        }

        // compute interior temperature
        float T_top_in = 0.0;
        for (int l = 0; l < 27; l++) {
            T_top_in += g[l * N + idx_top];
        }

        // impose heat flux boundary condition for temperature (q = -k * dT/dx)
        float T_top = T_top_in - q_wall / kappa; // dx = 1

        // adjust incoming temperature distribution to impose Neumann BC
        for (int l = 0; l < 27; l++) {
            if (d_cy[l] == -1) {
                float cu = d_cx[l] * u_lid; // since v = 0, w = 0 at the top wall
                float geq_top = d_weights[l] * T_top * (1.0 + 3.0 * cu); // first order accurate equilibrium for temperature at the wall
                g[l * N + idx] = -g[d_opp[l] * N + idx] + 2.0 * geq_top; // anti-bounce back for temperature to impose Neumann BC
            }
        }
    }
}

// Residual kernels for convergence monitoring
// Check criterion (||u_new - u_old|| / ||u_new|| < tol) at each lattice point
__global__ void lbm_u_residual_kernel(const float* __restrict__ f_old,
                                    const float* __restrict__ f_new,
                                    float* diff_acc, float* mag_acc,
                                    int nx, int ny, int nz)
{
    int N = nx * ny * nz; // total number of lattice points
    
    // -------- 3D grid and block configuration --------
    // define 3D global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return; // boundary check for thread space

    int idx = i + j * nx + k * nx * ny; // current index

    // -------- 2D grid and block configuration --------
    // define 2D global indices (i, j) 
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    // if (i >= nx || j >= ny) return; // boundary check for thread space

    // loop over k dimension
    // for (int k = 0; k < nz; k++) {
    //      int idx = i + j * nx + k * nx * ny;} // wrap the k loop till the end of the kernel

    // -------- 1D grid and block configuration --------
    // define global 1D linear index
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check for the 1D thread space
    // if (idx >= N) return;

    // calculate velocity from distribution functions
    float rho_old = 0.0, u_old = 0.0, v_old = 0.0, w_old = 0.0;
    float rho_new = 0.0, u_new = 0.0, v_new = 0.0, w_new = 0.0;

    #pragma unroll
    for (int l =0; l < 27; l++){
        float f_old_val = f_old[l * N + idx];
        float f_new_val = f_new[l * N + idx];
        rho_old += f_old_val;
        u_old += f_old_val * d_cx[l];
        v_old += f_old_val * d_cy[l];
        w_old += f_old_val * d_cz[l];

        rho_new += f_new_val;
        u_new += f_new_val * d_cx[l];
        v_new += f_new_val * d_cy[l];
        w_new += f_new_val * d_cz[l];
    }
    if (rho_old > 1e-9) {u_old /= rho_old; v_old /=rho_old; w_old /= rho_old;}
    else {u_old = 0.0; v_old = 0.0; w_old = 0.0;}

    if (rho_new > 1e-9) {u_new /= rho_new; v_new /= rho_new; w_new /= rho_new;}
    else {u_new = 0.0; v_new = 0.0; w_new = 0.0;}

    // calculate local residual
    float du = u_new - u_old; float dv = v_new - v_old; float dw = w_new - w_old;
    float diff_sq = du * du + dv * dv + dw * dw;
    float mag_sq = u_new * u_new + v_new * v_new + w_new * w_new;

    // atomic add to accumulate residual across threads
    atomicAdd(diff_acc, diff_sq); // accumulate sum of squared differences
    atomicAdd(mag_acc, mag_sq); // accumulate sum of squared magnitudes
}  

// Check criterion (||T_new - T_old|| / ||T_new|| < tol) at each lattice point
__global__ void lbm_T_residual_kernel(const float* __restrict__ g_old,
                                    const float* __restrict__ g_new,
                                    float* diff_acc, float* mag_acc,
                                    int nx, int ny, int nz)
{
    int N = nx * ny * nz; // total number of lattice points
    
    // -------- 3D grid and block configuration --------
    // define 3D global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return; // boundary check for thread space

    int idx = i + j * nx + k * nx * ny; // current index

    // -------- 2D grid and block configuration --------
    // define 2D global indices (i, j) 
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    // if (i >= nx || j >= ny) return; // boundary check for thread space

    // loop over k dimension
    // for (int k = 0; k < nz; k++) {
    //      int idx = i + j * nx + k * nx * ny;} // wrap the k loop till the end of the kernel

    // -------- 1D grid and block configuration --------
    // define global 1D linear index
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check for the 1D thread space
    // if (idx >= N) return;

    // calculate velocity from distribution functions
    float T_old = 0.0;
    float T_new = 0.0;

    #pragma unroll
    for (int l =0; l < 27; l++){
        float g_old_val = g_old[l * N + idx];
        float g_new_val = g_new[l * N + idx];

        T_old += g_old_val;
        T_new += g_new_val;
    }

    // calculate local residual
    float dT = T_new - T_old;
    float diff_sq = dT * dT;
    float mag_sq = T_new * T_new;

    // atomic add to accumulate residual across threads
    atomicAdd(diff_acc, diff_sq); // accumulate sum of squared differences
    atomicAdd(mag_acc, mag_sq); // accumulate sum of squared magnitudes
}

// ----- Host functions to launch kernels -----
// function to initialize GPU with parameters
void lbm_init_gpu(int nx, int ny, int nz,
                  int* h_cx, int* h_cy, int* h_cz, int* h_opp, float* h_weights)
{
    size_t size = 27 * nx * ny * nz * sizeof(float);
    // Allocate persistent device memory for distribution functions
    cudaMalloc((void**)&d_f_in, size);
    cudaMalloc((void**)&d_f_out, size);
    cudaMalloc((void**)&d_g_in, size);
    cudaMalloc((void**)&d_g_out, size);
    // Allocate device memory for residual
    cudaMalloc((void**)&d_u_diff_sq, sizeof(float));
    cudaMalloc((void**)&d_u_mag_sq, sizeof(float));
    cudaMalloc((void**)&d_T_diff_sq, sizeof(float));
    cudaMalloc((void**)&d_T_mag_sq, sizeof(float));
    // Copy D3Q27 parameters to GPU constant memory
    cudaMemcpyToSymbol(d_cx, h_cx, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, h_cy, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_cz, h_cz, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_opp, h_opp, 27 * sizeof(int));
    cudaMemcpyToSymbol(d_weights, h_weights, 27 * sizeof(float));
}

// function to copy initial conditons to GPU
void lbm_copy_host_to_device(float* h_f_in, float* h_g_in, int nx, int ny, int nz)
{
    size_t size = 27 * nx * ny * nz * sizeof(float);
    cudaMemcpy(d_f_in, h_f_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_in, h_g_in, size, cudaMemcpyHostToDevice);
}

// function to perform one LBM step
void lbm_run_step_gpu(int nx, int ny, int nz, float omega_f, float omega_g, float u_lid, float beta, float gravity, float T_ref,
                      float q_wall, float T_wall, float kappa)
{
    // define block and grid sizes
    // -------- 3D grid and block configuration --------
    dim3 block(8, 4, 4); // better occupancy with 1D flattened thread blocks
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    // -------- 2D grid and block configuration --------
    // dim3 block(16, 8, 1);
    // dim3 grid((nx + block.x -1) / block.x,
    //           (ny + block.y -1) / block.y, 1);

    // -------- 1D grid and block configuration --------
    // dim3 block(128, 1, 1); 
    // int N = nx * ny * nz;
    // dim3 grid((N + block.x - 1) / block.x, 1, 1);

    // launch bulk lbm kernel
    lbm_kernel_soa<<<grid, block>>>(d_f_in, d_f_out, d_g_in, d_g_out, nx, ny, nz, omega_f, omega_g, beta, gravity, T_ref);
    // launch boundary condition kernel
    apply_bc_kernel<<<grid, block>>>(d_f_out, d_g_out, nx, ny, nz, u_lid, q_wall, T_wall, kappa);
    // synchronize
    cudaDeviceSynchronize();
    // swap pointers for next iteration
    float* temp_f = d_f_in;
    d_f_in = d_f_out;
    d_f_out = temp_f;
    float* temp_g = d_g_in;
    d_g_in = d_g_out;
    d_g_out = temp_g;
}

// function to compute u residual for convergence monitoring
float lbm_compute_u_residual_gpu(int nx, int ny, int nz)  
{   
    // reset residual accumulators
    cudaMemset(d_u_diff_sq, 0, sizeof(float));
    cudaMemset(d_u_mag_sq, 0, sizeof(float));

    // define block and grid sizes
    // -------- 3D grid and block configuration --------
    dim3 block(8, 4, 4); // better occupancy with 1D flattened thread blocks
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    // -------- 2D grid and block configuration --------
    // dim3 block(16, 8, 1);
    // dim3 grid((nx + block.x -1) / block.x,
    //           (ny + block.y -1) / block.y, 1);
    
    // -------- 1D grid and block configuration --------
    // dim3 block(128, 1, 1); 
    // int N = nx * ny * nz;
    // dim3 grid((N + block.x - 1) / block.x, 1, 1);


    // launch residual kernel
    lbm_u_residual_kernel<<<grid, block>>>(d_f_in, d_f_out, d_u_diff_sq, d_u_mag_sq, nx, ny, nz);
    // synchronize
    cudaDeviceSynchronize();
    // copy results back to host and compute final residual
    float h_diff_sq, h_mag_sq;
    cudaMemcpy(&h_diff_sq, d_u_diff_sq, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_mag_sq, d_u_mag_sq, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_mag_sq < 1e-9) return 1.0; // avoid division by zero
    return sqrt(h_diff_sq) / sqrt(h_mag_sq);
}

// function to compute T residual for convergence monitoring
float lbm_compute_T_residual_gpu(int nx, int ny, int nz)  
{   
    // reset residual accumulators
    cudaMemset(d_T_diff_sq, 0, sizeof(float));
    cudaMemset(d_T_mag_sq, 0, sizeof(float));

    // define block and grid sizes
    // -------- 3D grid and block configuration --------
    dim3 block(8, 4, 4); // better occupancy with 1D flattened thread blocks
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    // -------- 2D grid and block configuration --------
    // dim3 block(16, 8, 1);
    // dim3 grid((nx + block.x -1) / block.x,
    //           (ny + block.y -1) / block.y, 1);
    
    // -------- 1D grid and block configuration --------
    // dim3 block(128, 1, 1); 
    // int N = nx * ny * nz;
    // dim3 grid((N + block.x - 1) / block.x, 1, 1);


    // launch residual kernel
    lbm_T_residual_kernel<<<grid, block>>>(d_g_in, d_g_out, d_T_diff_sq, d_T_mag_sq, nx, ny, nz);
    // synchronize
    cudaDeviceSynchronize();
    // copy results back to host and compute final residual
    float h_T_diff_sq, h_T_mag_sq;
    cudaMemcpy(&h_T_diff_sq, d_T_diff_sq, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_T_mag_sq, d_T_mag_sq, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_T_mag_sq < 1e-9) return 1.0; // avoid division by zero
    return sqrt(h_T_diff_sq) / sqrt(h_T_mag_sq);
}

// function to copy results back to host
void lbm_copy_device_to_host(float* h_f_out, float* h_g_out, int nx, int ny, int nz)
{
    size_t size = 27 * nx * ny * nz * sizeof(float);
    cudaMemcpy(h_f_out, d_f_in, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_out, d_g_in, size, cudaMemcpyDeviceToHost);
}

// function to free GPU memory
void lbm_free_gpu()
{
    if (d_f_in) cudaFree(d_f_in);
    if (d_f_out) cudaFree(d_f_out);
    if (d_g_in) cudaFree(d_g_in);
    if (d_g_out) cudaFree(d_g_out);
    if (d_u_diff_sq) cudaFree(d_u_diff_sq);
    if (d_u_mag_sq) cudaFree(d_u_mag_sq);
    if (d_T_diff_sq) cudaFree(d_T_diff_sq);
    if (d_T_mag_sq) cudaFree(d_T_mag_sq);
}





