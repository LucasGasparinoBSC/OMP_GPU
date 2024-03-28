#include "vecAdd.h"

void vecAddOMP_CPU(const float *A, const float *B, float *C, const uint64_t N)
{
    nvtxRangePush("VecAdd_OMP_CPU");
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
    nvtxRangePop();
}

void vecAddOMP_GPU(const float *A, const float *B, float *C, const uint64_t N)
{
    nvtxRangePush("VecAdd_OMP_GPU");
    #pragma omp target teams distribute parallel for
    for (uint64_t i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
    nvtxRangePop();
}

void vecAddACC_GPU(const float *A, const float *B, float *C, const uint64_t N)
{
    nvtxRangePush("VecAdd_ACC_GPU");
    #pragma acc parallel loop present(A, B, C)
    for (uint64_t i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
    nvtxRangePop();
}