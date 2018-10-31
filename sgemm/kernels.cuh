#ifndef __GG43j7hdVFHUret__
#define __GG43j7hdVFHUret__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void cuda_kernel_warmup(float *p)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float f = (float)idx;
    p[idx] = f * f * f;
}

__global__ void cuda_kernel_sgemm(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = idx / N;
    int n = idx % N;
#define idx(ri, ci, nc) ((ri) * (nc) + (ci))
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
    {
        acc += a[idx(m, k, K)] * b[idx(k, n, N)];
    }
    c[idx(m, n, N)] = alpha * acc + beta * c[idx(m, n, N)];
#undef idx
}

#endif