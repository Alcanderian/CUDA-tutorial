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
    size_t n, size_t m, size_t k)
{
}

#endif