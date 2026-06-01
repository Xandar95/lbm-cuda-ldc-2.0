#ifndef LBM_KERNEL_CUH
#define LBM_KERNEL_CUH

#ifdef __cplusplus
extern "C" {
#endif

void lbm_init_gpu(int nx, int ny, int nz,
                  int* h_cx, int* h_cy, int* h_cz, int* h_opp, float* h_weights);

void lbm_copy_host_to_device(float* h_f_in, float* h_g_in, 
                            int nx, int ny, int nz);

void lbm_run_step_gpu(int nx, int ny, int nz, float omega_f, float omega_g, float u_lid, float beta, float gravity, float T_ref,
                      float q_wall, float T_wall, float kappa);

float lbm_compute_u_residual_gpu(int nx, int ny, int nz) ;

float lbm_compute_T_residual_gpu(int nx, int ny, int nz) ;

void lbm_copy_device_to_host(float* h_f_out, float* h_g_out, 
                            int nx, int ny, int nz);

void lbm_free_gpu();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LBM_KERNEL_CUH

