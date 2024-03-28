#include "dotProduct.h"

void dotProductOMP_CPU(const float *A, const float *B, float C, const uint64_t N)
{
    C = 0.0f;
    nvtxRangePush("dotProduct_OMP_CPU");
    #pragma omp parallel for reduction(+:C)
    for (uint64_t i = 0; i < N; i++)
    {
        C += A[i] * B[i];
    }
    nvtxRangePop();
}

void dotProductOMP_GPU(const float *A, const float *B, float C, const uint64_t N)
{
    C = 0.0f;
    #pragma omp target update to(C)
    nvtxRangePush("dotProduct_OMP_GPU");
    #pragma omp target teams distribute parallel for is_device_ptr(A, B) reduction(+:C)
    for (uint64_t i = 0; i < N; i++)
    {
        C += A[i] * B[i];
    }
    nvtxRangePop();
}

void dotProductACC_GPU(const float *A, const float *B, float &C, const uint64_t N)
{
    C = 0.0f;
    #pragma acc update device (C)
    nvtxRangePush("dotProduct_ACC");
    #pragma acc parallel loop present(A, B, C) reduction(+:C)
    for (uint64_t i = 0; i < N; i++)
    {
        C += A[i] * B[i];
    }
    nvtxRangePop();
}