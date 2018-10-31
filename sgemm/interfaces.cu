#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "interfaces.h"
#include "kernels.cuh"
#include "../prof.h"

void gpu_sgemm(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    hs_timer timer;
    timer.tic("gpu sgemm");

    cudaMalloc((void **)&dev_a, M * K * sizeof(float));
    cudaMalloc((void **)&dev_b, K * N * sizeof(float));
    cudaMalloc((void **)&dev_c, M * N * sizeof(float));

    cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    cuda_kernel_sgemm<<<M, N>>>(dev_a, dev_b, dev_c, N, M, K, alpha, beta);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    timer.toc("gpu sgemm");
}

void gpu_warmup()
{
    float *dev_p = 0;

    hs_timer timer;
    timer.tic("gpu warmup");

    cudaMalloc((void **)&dev_p, 16 * 32 * sizeof(float));

    cuda_kernel_warmup<<<16, 32>>>(dev_p);

    cudaDeviceSynchronize();

    cudaFree(dev_p);

    timer.toc("gpu warmup");
}

void cpu_sgemm(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta)
{
    hs_timer timer;
    timer.tic("cpu sgemm");

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
    timer.toc("cpu sgemm");
}

void cpu_warmup()
{
    hs_timer timer;
    timer.tic("cpu warmup");

    const size_t arr_size = 1024;
    float *p = new float[arr_size];

#pragma omp parallel for simd
    for (size_t i = 0; i < arr_size; i++)
    {
        float f = (float)i;
        p[i] = f * f * f;
    }

    delete p;

    timer.toc("cpu warmup");
}
