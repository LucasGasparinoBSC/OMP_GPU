#include "vecAdd.h"
#include "dotProduct.h"
#include "matVec.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, const char** argv) {

    // Test dot product GPU kernels

    const uint64_t n = 1 << 20;
    printf("n = %lu\n", n);

    const u_int64_t niter = 101;

    float *A = (float*)malloc(n * sizeof(float));
    float *B = (float*)malloc(n * sizeof(float));
    float C  = 0.0f;

    for (uint64_t i = 0; i < n; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    // ACC version
    #pragma acc enter data copyin(A[0:n], B[0:n], C)
    for (uint64_t iter = 0; iter < niter; iter++) {
        dotProductACC_GPU(A, B, C, n);
    }
    #pragma acc exit data copyout(C)
    printf("C = %f\n", C);

    // OMP version
    #pragma omp target enter data map(to: A[0:n], B[0:n], C)
    for (uint64_t iter = 0; iter < niter; iter++) {
        dotProductOMP_GPU(A, B, C, n);
    }
    #pragma omp target exit data map(from: C)
    printf("C = %f\n", C);

    return 0;
}