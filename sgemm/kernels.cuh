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

// naive!!
__global__ void cuda_kernel_sgemm_0(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta)
{
    int ir = blockIdx.x * 32 + threadIdx.x; // row idx in global
    int ic = blockIdx.y * 32 + threadIdx.y; // col idx in global

    if (ir < M && ic < N)
    {
#define idx(ri, ci, nc) ((ri) * (nc) + (ci))
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            acc += a[idx(ir, k, K)] * b[idx(k, ic, N)];
        }
        c[idx(ir, ic, N)] = alpha * acc + beta * c[idx(ir, ic, N)];
#undef idx
    }
}

// use shared memory & tile
__global__ void cuda_kernel_sgemm_1(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta)
{
    int tr = threadIdx.x;                   // row idx in block
    int tc = threadIdx.y;                   // col idx in block
    int ir = blockIdx.x * 32 + threadIdx.x; // row idx in global
    int ic = blockIdx.y * 32 + threadIdx.y; // col idx in global

    __shared__ float a_sub[32][32];
    __shared__ float b_sub[32][32];

    int load_size = K / 32;
    if (K % 32 != 0)
    {
        load_size += 1;
    }
    float acc = 0.0f;
    int a_ir = ir;
    int b_ic = ic;
#define idx(ri, ci, nc) ((ri) * (nc) + (ci))
    for (int l = 0; l < load_size; ++l)
    {
        int a_ic = l * 32 + tc;
        int b_ir = l * 32 + tr;
        a_sub[tr][tc] = 0.0f;
        b_sub[tr][tc] = 0.0f;
        if (a_ir < M && a_ic < K)
            a_sub[tr][tc] = a[idx(a_ir, a_ic, K)];
        if (b_ir < K && b_ic < N)
            b_sub[tr][tc] = b[idx(b_ir, b_ic, N)];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < 32; ++k)
        {
            acc += a_sub[tr][k] * b_sub[k][tc];
        }

        __syncthreads();
    }

    if (ir < M && ic < N)
        c[idx(ir, ic, N)] = alpha * acc + beta * c[idx(ir, ic, N)];
#undef idx
}

// use __ldg & avoid bank conflict
__global__ void cuda_kernel_sgemm_2(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta)
{
    int tr = threadIdx.x;                   // row idx in block
    int tc = threadIdx.y;                   // col idx in block
    int ir = blockIdx.x * 32 + threadIdx.x; // row idx in global
    int ic = blockIdx.y * 32 + threadIdx.y; // col idx in global

    __shared__ float a_sub[32][32 + 1]; // avoid bank conflict
    __shared__ float b_sub[32][32 + 1];

    int load_size = K / 32;
    if (K % 32 != 0)
    {
        load_size += 1;
    }
    float acc = 0.0f;
    int a_ir = ir;
    int b_ic = ic;
#define idx(ri, ci, nc) ((ri) * (nc) + (ci))
    for (int l = 0; l < load_size; ++l)
    {
        int a_ic = l * 32 + tc;
        int b_ir = l * 32 + tr;
        a_sub[tr][tc] = 0.0f;
        b_sub[tr][tc] = 0.0f;
        if (a_ir < M && a_ic < K)
            a_sub[tr][tc] = __ldg(&a[idx(a_ir, a_ic, K)]); // cache
        if (b_ir < K && b_ic < N)
            b_sub[tr][tc] = __ldg(&b[idx(b_ir, b_ic, N)]);

        __syncthreads();
        
#pragma unroll
        for (int k = 0; k < 32; ++k)
        {
            acc += a_sub[tr][k] * b_sub[k][tc];
        }

        __syncthreads();
    }

    if (ir < M && ic < N)
        c[idx(ir, ic, N)] = alpha * acc + beta * c[idx(ir, ic, N)];
#undef idx
}

#endif