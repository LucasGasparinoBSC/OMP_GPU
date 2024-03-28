#include "matVec.h"

void matVecOMP(const uint64_t N, const float* A, const float* x, float* y) {
    nvtxRangePush("matVec_OMP_GPU");
    #pragma omp target teams distribute parallel for
    for (uint64_t i = 0; i < N; i++) {
        y[i] = 0.0f;
        for (uint64_t j = 0; j < N; j++) {
            y[i] += A[i*N + j] * x[j];
        }
    }
    nvtxRangePop();
}

void matVecACC(const uint64_t N, const float* A, const float* x, float* y) {
    nvtxRangePush("matVec_ACC");
    #pragma acc parallel loop gang vector
    for (uint64_t i = 0; i < N; i++) {
        y[i] = 0.0f;
        for (uint64_t j = 0; j < N; j++) {
            y[i] += A[i*N + j] * x[j];
        }
    }
    nvtxRangePop();
}

__global__ void matVecCUDA(const uint64_t N, const float* A, const float* x, float* y) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = 0.0f;
        for (uint64_t j = 0; j < N; j++) {
            y[i] += A[i*N + j] * x[j];
        }
    }
}