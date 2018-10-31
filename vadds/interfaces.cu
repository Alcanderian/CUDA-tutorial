#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include "interfaces.h"
#include "kernels.cuh"

void run_vadds(
    float *a, float *b, float *c, size_t arr_size,
    size_t grid_x, size_t block_x)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    printf("[vadds start]\n");

    size_t load_size = arr_size / (grid_x * block_x);
    if (load_size * grid_x * block_x != arr_size)
        load_size += 1;
    size_t tot_size = load_size * grid_x * block_x;

    printf("[vadds alloc]: arr_size= %llu, load_size= %llu, tot_size= %llu, threads= %llu\n", arr_size, load_size, tot_size, grid_x * block_x);

    cudaMalloc((void **)&dev_a, tot_size * sizeof(float));
    cudaMalloc((void **)&dev_b, tot_size * sizeof(float));
    cudaMalloc((void **)&dev_c, tot_size * sizeof(float));
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("[runtime error][%s %s]: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));
        return;
    }

    cudaMemcpy(dev_a, a, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("[runtime error][%s %s]: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));
        return;
    }

    dim3 grid_size(grid_x, 1, 1);
    dim3 block_size(block_x, 1, 1);
    cuda_kernel_vadds<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, load_size);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("[runtime error][%s %s]: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));
        return;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(c, dev_c, arr_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("[runtime error][%s %s]: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));
        return;
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    printf("[vadds finish]\n");
}

void verify_vadds(float *a, float *b, float *c, size_t arr_size)
{
    printf("[vadds verifying]\n");
    int pass = 1;
    for (size_t i = 0; i < arr_size; ++i)
    {
        if (fabs(a[i] + b[i] - c[i]) > 1e-7)
        {
            pass = 0;
            printf("[vadds wrong answer]: at= %llu, std= %f\t, ans= %f\t\n", i, a[i] + b[i], c[i]);
            break;
        }
    }
    if (pass)
    {
        printf("[vadds accept]\n");
    }
}