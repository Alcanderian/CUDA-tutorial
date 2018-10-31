#ifndef __GG43j7hdVFHUret__
#define __GG43j7hdVFHUret__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void cuda_kernel_vadds(float *a, float *b, float *c, size_t load_size)
{
    size_t load_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float *_a = a + load_size * load_idx;
    float *_b = b + load_size * load_idx;
    float *_c = c + load_size * load_idx;

    for (size_t i = 0; i < load_size; ++i)
    {
        _c[i] = _a[i] + _b[i];
    }
}

#endif