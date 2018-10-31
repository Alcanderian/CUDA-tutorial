#ifndef __ad93IFM09mf__
#define __ad93IFM09mf__

#include <stdlib.h>

void gpu_sgemm(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta, int kernel_type);
void gpu_warmup();
void cpu_sgemm(
    float *a, float *b, float *c,
    size_t N, size_t M, size_t K,
    float alpha, float beta);
void cpu_warmup();
#endif