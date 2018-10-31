#ifndef __fHDId9q2KID9__
#define __fHDId9q2KID9__

#include <stdlib.h>

void cpu_kernel_sgemm_0(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta)
{
#define idx(ri, ci, nc) ((ri) * (nc) + (ci))
    float *bt = new float[K * N];
#pragma omp parallel for simd
    for (int n = 0; n < N; ++n)
    {
        for (int k = 0; k < K; ++k)
        {
            bt[idx(n, k, K)] = b[idx(k, n, N)];
        }
    }
#pragma omp parallel for simd
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                acc += a[idx(m, k, K)] * bt[idx(n, k, K)];
            }
            c[idx(m, n, N)] = alpha * acc + beta * c[idx(m, n, N)];
        }
    }
    delete bt;
#undef idx
}

#endif