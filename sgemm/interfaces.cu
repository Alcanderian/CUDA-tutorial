#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mkl_cblas.h>

#include "interfaces.h"
#include "kernels.cuh"
#include "kernels.h"
#include "../prof.h"

void gpu_sgemm(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta, int kernel_type)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    cublasHandle_t handle;

    hs_timer timer;
    timer.tic("gpu sgemm");

    if (kernel_type == 'b')
        cublasCreate(&handle);

    cudaMalloc((void **)&dev_a, M * K * sizeof(float));
    cudaMalloc((void **)&dev_b, K * N * sizeof(float));
    cudaMalloc((void **)&dev_c, M * N * sizeof(float));

    cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, M * N * sizeof(float), cudaMemcpyHostToDevice);

    switch (kernel_type)
    {
    case 0:
    {
        int grid_r = M / 32;
        int grid_c = N / 32;
        if (M % 32 != 0)
            grid_r += 1;
        if (N % 32 != 0)
            grid_c += 1;
        dim3 grid_d(grid_r, grid_c, 1);
        dim3 block_d(32, 32, 1);
        cuda_kernel_sgemm_0<<<grid_d, block_d>>>(dev_a, dev_b, dev_c, N, M, K, alpha, beta);
        break;
    }
    case 1:
    {
        int grid_r = M / 32;
        int grid_c = N / 32;
        if (M % 32 != 0)
            grid_r += 1;
        if (N % 32 != 0)
            grid_c += 1;
        dim3 grid_d(grid_r, grid_c, 1);
        dim3 block_d(32, 32, 1);
        cuda_kernel_sgemm_1<<<grid_d, block_d>>>(dev_a, dev_b, dev_c, N, M, K, alpha, beta);
        break;
    }
    case 'b':
    {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dev_b, N, dev_a, K, &beta, dev_c, N);
        break;
    }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (kernel_type == 'b')
        cublasDestroy(handle);

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
    float alpha, float beta, int kernel_type)
{
    hs_timer timer;
    timer.tic("cpu sgemm");

    switch (kernel_type)
    {
    case 0:
    {
        cpu_kernel_sgemm_0(a, b, c, N, M, K, alpha, beta);
        break;
    }
    case 'm':
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, M, K, alpha, b, N, a, K, beta, c, N);
        break;
    }
    }
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
