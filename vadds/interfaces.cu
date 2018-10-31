#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "interfaces.h"
#include "kernels.cuh"
#include "../prof.h"

void gpu_vadds(
    float *a, float *b, float *c, size_t arr_size,
    size_t grid_x, size_t block_x)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    size_t load_size = arr_size / (grid_x * block_x);
    if (load_size * grid_x * block_x != arr_size)
        load_size += 1;
    size_t tot_size = load_size * grid_x * block_x;

    hs_timer timer;
    timer.tic("gpu vadds");

    cudaMalloc((void **)&dev_a, tot_size * sizeof(float));
    cudaMalloc((void **)&dev_b, tot_size * sizeof(float));
    cudaMalloc((void **)&dev_c, tot_size * sizeof(float));

    cudaMemcpy(dev_a, a, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arr_size * sizeof(float), cudaMemcpyHostToDevice);

    cuda_kernel_vadds<<<grid_x, block_x>>>(dev_a, dev_b, dev_c, load_size);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, arr_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    timer.toc("gpu vadds");
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

void cpu_vadds(float *a, float *b, float *c, size_t arr_size)
{
    hs_timer timer;
    timer.tic("cpu vadds");

#pragma omp parallel for simd
    for (size_t i = 0; i < arr_size; i++)
    {
        c[i] = a[i] + b[i];
    }

    timer.toc("cpu vadds");
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
