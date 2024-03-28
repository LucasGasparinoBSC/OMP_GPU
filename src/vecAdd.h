#ifndef VECADD_H_
#define VECADD_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvToolsExt.h>
#include <openacc.h>
#include <omp.h>
#include <cstdint>

// OMP CPU version
void vecAddOMP_CPU(const float *A, const float *B, float *C, const uint64_t N);

// OMP GPU version
void vecAddOMP_GPU(const float *A, const float *B, float *C, const uint64_t N);

// ACC version
void vecAddACC_GPU(const float *A, const float *B, float *C, const uint64_t N);

#endif // !VECADD_H_