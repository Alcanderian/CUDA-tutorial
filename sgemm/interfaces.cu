#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "interfaces.h"
#include "kernels.cuh"
#include "../prof.h"

void gpu_sgemm(float *a, float *b, float *c, size_t n, size_t m, size_t k)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    hs_timer timer;
    timer.tic("gpu sgemm");

    cudaMalloc((void **)&dev_a, n * m * sizeof(float));
    cudaMalloc((void **)&dev_b, m * k * sizeof(float));
    cudaMalloc((void **)&dev_c, n * k * sizeof(float));

    cudaMemcpy(dev_a, a, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, m * k * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, n * k * sizeof(float), cudaMemcpyDeviceToHost);

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

void cpu_sgemm(float *a, float *b, float *c, size_t n, size_t m, size_t k)
{
    hs_timer timer;
    timer.tic("cpu sgemm");

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

