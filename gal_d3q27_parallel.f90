program gal_kettle_d3q27_parallel
    ! Heated Lid driven cavity flow simulation using D3Q27 Lattice Boltzmann Method with parallel computing
    use iso_c_binding
    implicit none

    ! Parameters
    integer(c_int), parameter :: nx = 126, ny = 126, nz = 126 ! grid dimensions
    integer :: nstep = 1000000 ! number of time steps (choose large enough for convergence)
    integer :: i, j, k, l, nt
    real(c_float) :: xl, yl, zl, dx, dy, dz, nu, alpha, beta, Re, c2, omega_f, omega_g, u_lid, q_wall, q_top, kappa_wall, kappa_top, T_ref, gravity, cu, u2, residual_u, residual_T
    integer :: count_start, count_end, count_rate ! for timing

    ! Variable arrays
    real(c_float), dimension(nx, ny, nz) :: rho, T, u, v, w, x, y, z
    real(c_float), target, dimension(nx, ny, nz, 0:26) :: f, g ! distribution functions for momentum and temperature

    ! Constant arrays
    real(c_float), target, dimension(0:26) :: weights
    integer(c_int), target, dimension(0:26) :: cx, cy, cz, opp

    ! Variables for Binary VTK Output (For visualization in Paraview)
    integer(c_int32_t) :: n_points, p_bytes, T_bytes, v_bytes, offset_v, offset_T
    real(c_float), allocatable :: p_out(:), T_out(:), v_out(:,:)
    character(len=256) :: filename
    character(len=1024) :: buffer
    character(len=1) :: nl = new_line('a')
    integer :: idx
    real(c_float) :: rho_loc, u_loc, v_loc, w_loc, p_loc, T_loc

    ! --- Interfaces to CUDA kernel ---
    interface
        subroutine lbm_init_gpu(nx, ny, nz, cx, cy, cz, opp, weights) bind(C, name="lbm_init_gpu")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            integer(c_int), intent(in) :: cx(*), cy(*), cz(*), opp(*)
            real(c_float), intent(in) :: weights(*)
        end subroutine lbm_init_gpu

        subroutine lbm_copy_host_to_device(f, g, nx, ny, nz) bind(C, name="lbm_copy_host_to_device")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            real(c_float), intent(in) :: f(*) ! read only
            real(c_float), intent(in) :: g(*) ! read only
        end subroutine lbm_copy_host_to_device

        subroutine lbm_run_step_gpu(nx, ny, nz, omega_f, omega_g, u_lid, beta, gravity, T_ref, q_wall, q_top, kappa_wall, kappa_top) bind(C, name="lbm_run_step_gpu")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            real(c_float), value :: omega_f, omega_g, u_lid, beta, gravity, T_ref, q_wall, q_top, kappa_wall, kappa_top
        end subroutine lbm_run_step_gpu

        function lbm_compute_u_residual_gpu(nx, ny, nz) bind(C, name="lbm_compute_u_residual_gpu")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            real(c_float) :: lbm_compute_u_residual_gpu
        end function lbm_compute_u_residual_gpu

        function lbm_compute_T_residual_gpu(nx, ny, nz) bind(C, name="lbm_compute_T_residual_gpu")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            real(c_float) :: lbm_compute_T_residual_gpu
        end function lbm_compute_T_residual_gpu

        subroutine lbm_copy_device_to_host(f, g, nx, ny, nz) bind(C, name="lbm_copy_device_to_host")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            real(c_float), intent(out) :: f(*) ! write only
            real(c_float), intent(out) :: g(*) ! write only
        end subroutine lbm_copy_device_to_host

        subroutine lbm_free_gpu() bind(C, name="lbm_free_gpu")
            use iso_c_binding
        end subroutine lbm_free_gpu
    end interface
    
    call system_clock(count_start, count_rate) ! Start simulation timing
    call system_clock(count_start)

    ! ---- Initialization ----
    xl = 1.0e0
    yl = 1.0e0
    zl = 1.0e0
    dx = xl / (nx - 1)
    dy = yl / (ny - 1)
    dz = zl / (nz - 1)
    c2 = 1.0e0 / 3.0e0 ! lattice speed of sound squared for D3Q27 (i.e., cs = c/sqrt(3))
    u_lid = 0.01e0 ! Lid velocity in the lattice
    T_ref = 0.0e0 ! Reference temperature for buoyancy (normalized)
    Re = 100.0e0 ! Desired Reynolds number
    nu = u_lid * (ny - 1.0e0) / Re ! Recalculate viscosity based on Re and lid velocity
    alpha = 0.1e0 ! Thermal diffusivity
    beta = 1.0e-4 ! Thermal expansion coefficient (for buoyancy)
    kappa_wall = 1.0e0 ! Thermal conductivity at the walls (for temperature BCs)
    kappa_top = 0.1e0 ! Thermal conductivity at the top boundary
    q_wall = 1.0e-4 ! Heat flux at the walls (normalized)
    q_top = -1.0e-5 ! Heat flux at the top boundary (normalized)
    gravity = 1.0e-3! Gravitational acceleration (normalized)
    omega_f = 1.0e0 / (3.0e0 * nu + 0.5e0) ! Relaxation parameter (SRT model)
    omega_g = 1.0e0 / (3.0e0 * alpha + 0.5e0) ! Relaxation parameter for temperature field
    ! Ma = u_lid / cs should be < 0.1 for incompressibility

    ! Lattice weights for D3Q27
    weights = [8.0e0/27.0e0, &                                                                             ! w0
               2.0e0/27.0e0, 2.0e0/27.0e0, 2.0e0/27.0e0, 2.0e0/27.0e0, 2.0e0/27.0e0, 2.0e0/27.0e0, &         ! w1-w6
               1.0e0/54.0e0, 1.0e0/54.0e0, 1.0e0/54.0e0, 1.0e0/54.0e0, 1.0e0/54.0e0, 1.0e0/54.0e0, &       ! w7-w18
               1.0e0/54.0e0, 1.0e0/54.0e0, 1.0e0/54.0e0, 1.0e0/54.0e0, 1.0e0/54.0e0, 1.0e0/54.0e0, &
               1.0e0/216.0e0, 1.0e0/216.0e0, 1.0e0/216.0e0, 1.0e0/216.0e0, 1.0e0/216.0e0, 1.0e0/216.0e0, & ! w19-w26
               1.0e0/216.0e0, 1.0e0/216.0e0]

    ! Lattice velocities for D3Q27
    cx = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1]
    cy = [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1] ! wall normal in y
    cz = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 0, 0, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1] 

    ! Opposite directions for bounce-back
    opp = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25]

    ! Set lattice positions
    x(1, :, :) = 0.0e0
    y(:, 1, :) = 0.0e0
    z(:, :, 1) = 0.0e0
    do i = 1, nx-1
        x(i+1, :, :) = x(i, :, :) + dx
    end do
    do j = 1, ny-1
        y(:, j+1, :) = y(:, j, :) + dy
    end do
    do k = 1, nz-1
        z(:, :, k+1) = z(:, :, k) + dz
    end do

    ! Initial condition: rho = 1, T = 0, u = 0, v = 0, w = 0
    rho = 1.0e0 ! Initial density (normalized)
    T = 1.0e0 ! Initial temperature (normalized)
    u = 0.0e0
    v = 0.0e0
    w = 0.0e0
    u(:, ny, :) = u_lid ! Set lid velocity at the top boundary
    ! T(1, :, :) = T_wall ! Set wall temperature at the west boundary
    ! T(nx, :, :) = T_wall ! Set wall temperature at the east boundary

    ! Initialize distributions (at equilibrium) (SOA layout)
    do l = 0, 26
        do k = 1, nz
            do j = 1, ny
                do i = 1, nx
                    cu = cx(l) * u(i, j, k) + cy(l) * v(i, j, k) + cz(l) * w(i, j, k)
                    u2 = u(i, j, k)**2 + v(i, j, k)**2 + w(i, j, k)**2
                    f(i, j, k, l) = weights(l) * rho(i, j, k) * (1.0e0 + 3.0e0 * cu + 4.5e0 * cu**2 - 1.5e0 * u2) ! 2nd order accurate equilibrium distribution function
                    g(i, j, k, l) = weights(l) * T(i, j, k) * (1.0e0 + 3.0e0 * cu) ! 1st order accurate equilibrium distribution function for temperature field
                end do
            end do
        end do
    end do

    ! Initialize GPU
    print *, 'Initializing GPU...'
    call lbm_init_gpu(nx, ny, nz, cx, cy, cz, opp, weights)

    ! Copy initial distribution to GPU
    print *, 'Copying initial data to GPU...'
    call lbm_copy_host_to_device(f, g, nx, ny, nz)

    ! Initialize Binary Output Buffers
    n_points = nx * ny * nz
    p_bytes = n_points * 4         ! 4 bytes per float (single precision)
    T_bytes = n_points * 4         ! 4 bytes per float (single precision)
    v_bytes = n_points * 3 * 4     ! 3 components * 4 bytes per float
    offset_T = p_bytes + 4         ! Offset for temperature data block (pressure size + 4 byte header)
    offset_v = offset_T + T_bytes + 4 ! Offset for velocity data block (temperature size + 4 byte header)
    
    allocate(p_out(n_points), T_out(n_points), v_out(3, n_points))

    ! ---- GPU Time loop ----
    print *, 'Starting time loop on GPU...'

    do nt = 1, nstep
        call lbm_run_step_gpu(nx, ny, nz, omega_f, omega_g, u_lid, beta, gravity, T_ref, q_wall, q_top, kappa_wall, kappa_top)
        
        ! progress output
        if (mod(nt, 1000) == 0) then ! Check residual every 1000 steps\
            
            residual_u = lbm_compute_u_residual_gpu(nx, ny, nz)
            residual_T = lbm_compute_T_residual_gpu(nx, ny, nz)
            print *, 'Time step: ', nt, ' Residual u: ', residual_u, ' Residual T: ', residual_T

            ! Copy results back to host
            call lbm_copy_device_to_host(f, g, nx, ny, nz)

            ! Compute variables and fill 1D binary buffers directly
            idx = 1
            do k = 1, nz
                do j = 1, ny
                    do i = 1, nx
                        rho_loc = 0.0e0; T_loc = 0.0e0; u_loc = 0.0e0; v_loc = 0.0e0; w_loc = 0.0e0
                        
                        do l = 0, 26
                            rho_loc = rho_loc + f(i, j, k, l)
                            T_loc = T_loc + g(i, j, k, l)
                            u_loc = u_loc + f(i, j, k, l) * cx(l)
                            v_loc = v_loc + f(i, j, k, l) * cy(l)
                            w_loc = w_loc + f(i, j, k, l) * cz(l)
                        end do
                        
                        u_loc = u_loc / rho_loc
                        v_loc = v_loc / rho_loc
                        w_loc = w_loc / rho_loc
                        p_loc = c2 * rho_loc 

                        p_out(idx) = p_loc
                        T_out(idx) = T_loc

                        v_out(1, idx) = u_loc
                        v_out(2, idx) = v_loc
                        v_out(3, idx) = w_loc
                        
                        idx = idx + 1
                    end do
                end do
            end do

             ! Generate filename and write Binary XML .vti file
            write(filename, '("ldc_D3Q27_", I6.6, ".vti")') nt
            
            ! access='stream' allows us to write raw, unformatted bytes
            open(unit=10, file=trim(filename), status='replace', access='stream', form='unformatted')
            
            ! -- Write XML Headers --
            write(buffer, '(A)') '<?xml version="1.0"?>'
            write(10) trim(buffer) // nl
            write(buffer, '(A)') '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt32">'
            write(10) trim(buffer) // nl
            
            ! Define Origin and Spacing (Note: VTK grid extents are zero-indexed)
            write(buffer, '("<ImageData WholeExtent=""0 ", I0, " 0 ", I0, " 0 ", I0, """ Origin=""0 0 0"" Spacing=""", ES14.6, " ", ES14.6, " ", ES14.6, """>")') nx-1, ny-1, nz-1, dx, dy, dz
            write(10) trim(buffer) // nl
            
            write(buffer, '("  <Piece Extent=""0 ", I0, " 0 ", I0, " 0 ", I0, """>")') nx-1, ny-1, nz-1
            write(10) trim(buffer) // nl
            write(buffer, '(A)') '    <PointData Scalars="pressure" Vectors="velocity">'
            write(10) trim(buffer) // nl
            
            ! Define Data Arrays and pointers to their binary offsets
            write(buffer, '(A)') '      <DataArray type="Float32" Name="pressure" format="appended" offset="0"/>'
            write(10) trim(buffer) // nl
            write(buffer, '("      <DataArray type=""Float32"" Name=""temperature"" format=""appended"" offset=""", I0, """/>")') offset_T
            write(10) trim(buffer) // nl
            write(buffer, '("      <DataArray type=""Float32"" Name=""velocity"" NumberOfComponents=""3"" format=""appended"" offset=""", I0, """/>")') offset_v
            write(10) trim(buffer) // nl
            
            write(buffer, '(A)') '    </PointData>'
            write(10) trim(buffer) // nl
            write(buffer, '(A)') '  </Piece>'
            write(10) trim(buffer) // nl
            write(buffer, '(A)') '</ImageData>'
            write(10) trim(buffer) // nl
            write(buffer, '(A)') '<AppendedData encoding="raw">'
            write(10) trim(buffer) // nl
            
            ! -- Write Raw Binary Data --
            write(10) '_'       ! VTK requires an underscore immediately before raw data
            write(10) p_bytes   ! 32-bit header indicating byte length of pressure array
            write(10) p_out     ! Raw bytes: Pressure 
            write(10) T_bytes   ! 32-bit header indicating byte length of temperature array
            write(10) T_out     ! Raw bytes: Temperature
            write(10) v_bytes   ! 32-bit header indicating byte length of velocity array
            write(10) v_out     ! Raw bytes: Velocity
            
            ! Close XML tags
            write(buffer, '(A)') '</AppendedData>'
            write(10) nl // trim(buffer) // nl
            write(buffer, '(A)') '</VTKFile>'
            write(10) trim(buffer) // nl
            
            close(10)
            
            if (residual_u < 1.0e-4 .and. residual_T < 1.0e-5) then
                print *, 'Simulation converged at time step: ', nt
                exit
            end if
        end if
    end do

    ! Free GPU resources
    call lbm_free_gpu()

    call system_clock(count_end) ! End simulation timing
    print *, 'Total simulation time (s): ', real(count_end - count_start) / real(count_rate)
    
end program gal_kettle_d3q27_parallel
