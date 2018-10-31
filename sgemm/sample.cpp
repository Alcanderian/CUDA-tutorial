#include <math.h>
#include <stdio.h>
#include <omp.h>

#include "interfaces.h"

void verify(float *a, float *b, size_t arr_size)
{
    int pass = 1;
    for (size_t i = 0; i < arr_size; ++i)
    {
        // relative error
        if (fabs((a[i] - b[i]) / a[i]) > 1e-6)
        {
            pass = 0;
            printf("[wrong answer]: at= %llu, a= %f\t, b= %f\t\n", i, a[i], b[i]);
            break;
        }
    }
}

int main()
{
    // dont use 2^n, it will cause cache crash
    const size_t N = 1000, M = 500, K = 5000;
    const float alpha = M_PI, beta = M_E;
    float *a = new float[M * K];
    float *b = new float[K * N];
    float *c1 = new float[M * N];
    float *c2 = new float[M * N];

#pragma omp parallel
    {
#pragma omp for
        for (size_t i = 0; i < M * K; ++i)
        {
            float f= (float)i;
            a[i] = cosf(f) * cosf(f);
        }

#pragma omp for
        for (size_t i = 0; i < K * N; ++i)
        {
            float f= (float)i;
            b[i] = sinf(f) * sinf(f);
        }
#pragma omp for
        for (size_t i = 0; i < M * N; ++i)
        {
            float f= (float)i;
            c1[i] = cosf(f) * sinf(f);
            c2[i] = cosf(f) * sinf(f);
        }
    }

    gpu_warmup();
    cpu_warmup();
    cpu_sgemm(a, b, c1, N, M, K, alpha, beta);
    gpu_sgemm(a, b, c2, N, M, K, alpha, beta);
    verify(c1, c2, M * N);

    delete a;
    delete b;
    delete c1;
    delete c2;
    return 0;
}