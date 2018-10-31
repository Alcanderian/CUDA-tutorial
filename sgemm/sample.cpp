#include <math.h>
#include <stdio.h>
#include <omp.h>

#include "interfaces.h"

void verify(float *a, float *b, size_t arr_size)
{
    int pass = 1;
    for (size_t i = 0; i < arr_size; ++i)
    {
        if (fabs(a[i] - b[i]) > 1e-7)
        {
            pass = 0;
            printf("[wrong answer]: at= %llu, a= %f\t, b= %f\t\n", i, a[i], b[i]);
            break;
        }
    }
}

int main()
{
    const size_t n = 4096, m = 2048, k = 4096;
    float *a = new float[n * m];
    float *b = new float[m * k];
    float *c1 = new float[n * k];
    float *c2 = new float[n * k];

#pragma omp parallel
    {
#pragma omp for
        for (size_t i = 0; i < n * m; ++i)
        {
            float f = (float)i;
            a[i] = sinf(f) * sinf(f);
        }

#pragma omp for
        for (size_t i = 0; i < m * k; ++i)
        {
            float f = (float)i;
            b[i] = cosf(f) * cosf(f);
        }
    }

    gpu_warmup();
    cpu_warmup();
    return 0;
}