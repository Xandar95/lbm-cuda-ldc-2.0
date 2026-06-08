# GPU-Accelerated 3D LBM Solver for Heated-Lid Driven Cavity Flow

This repository contains a GPU-accelerated implementation of a 3D Lattice Boltzmann Method (LBM) solver for the heated lid driven cavity flow with buoyancy/natural convection.

## Features
- D3Q27 lattice
- Single Relaxation Time (SRT) Bhatnagar-Gross-Krook (BGK) model
- Pull streaming on parallel code with Structure-of-Arrays (SoA) memory layout
- Flow characterized by \(Re, Pr, Ra\) numbers
- Only global memory and constant memory utilized
- Convergence criterion to break the time loop velocity & temperature change between 1000 time steps reach 10^-5 (adjustable threshold)
- 1D, 2D, and 3D grid/block configurations included

## Directory Structure
- src/ - CUDA kernels, interfaces, and host code
- figures/ - Post-processed velocity, temperature profiles and streamlines.
- post-process/ 
    - Post-processing script to obtain plane-wise and slice-wise velocity contours, streamlines, temperature contours, and pressure profiles.
    - Conversion script to convert .csv data files from the numerical solver to .vti format to be visualized in ParaView.

## Kernel Structure
-

## Build Requirements
- NVIDIA GPU with atleast 4GB VRAM
- CUDA Toolkit or NVIDIA HPC_SDK for nvcc compiler

## Build Instructions (On Linux/WSL)
- Update all dependencies 'sudo apt update', 'sudo apt upgrade'.
- Clone the repository with 'git clone '.
- Install CMake 'sudo apt install cmake'.
- Build using 'mkdir build', 'cd build', 'cmake ..', 'cmake --build .'.
- Edit 'CMakeLists.txt' if using 'g++' >= 12. Also change the GPU architecture (e.g '86' for Ampere) based on your GPU.
- The numerical solver results will be written to a separate sim_output folder.

## Post-process Instructions
- Install virtual environment package for Python 'python3 pip install venv'.
- Create a virtual environment 'python3 -m venv venv'.
- Activate virtual environment 'source ./venv/bin/activate'.
- Install the dependencies 'pip install -r requirements.txt'.
- Run the post processing script to get the plane-wise slice-wise contours 'python3 post_process.py'.
- Run the conversion script to convert the .csv data files to .vti format 'python3 convert_vti.py'.


