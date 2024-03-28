#ifndef MATVEC_H_
#define MATVEC_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvToolsExt.h>
#include <openacc.h>
#include <omp.h>
#include <cstdint>

// OMP GPU version
void matVecOMP(const uint64_t N, const float* A, const float* x, float* y);

// ACC version
void matVecACC(const uint64_t N, const float* A, const float* x, float* y);

// CUDA version
__global__ void matVecCUDA(const uint64_t N, const float* A, const float* x, float* y);

#endif // !MATVEC_H_