cmake_minimum_required(VERSION 3.20)

# set compiler CUDA host compiler to g++-12 (comment out if using a g++>=12 as default)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)

project(LBM-CUDA-LDC LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set GPU architecture (set to 86 for RTX 3000 series, adjust as needed for other GPUs)
set(CMAKE_CUDA_ARCHITECTURES 86) 

# create executable
add_executable(lbm_sim src/main.cpp src/lbm_kernel.cu)




