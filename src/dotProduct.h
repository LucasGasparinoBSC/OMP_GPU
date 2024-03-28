#ifndef DOTPRODUCT_H_
#define DOTPRODUCT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvToolsExt.h>
#include <openacc.h>
#include <omp.h>
#include <cstdint>

// OMP CPU version
void dotProductOMP_CPU(const float *A, const float *B, float C, const uint64_t N);

// OMP GPU version
void dotProductOMP_GPU(const float *A, const float *B, float C, const uint64_t N);

// ACC GPU version
void dotProductACC_GPU(const float *A, const float *B, float &C, const uint64_t N);

#endif // !DOTPRODUCT_H_