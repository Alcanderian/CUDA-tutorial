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
    const size_t arr_size = 500 * 1000 * 1000;
    float *a = new float[arr_size];
    float *b = new float[arr_size];
    float *c1 = new float[arr_size];
    float *c2 = new float[arr_size];

#pragma omp parallel for simd
    for (size_t i = 0; i < arr_size; ++i)
    {
        float f = (float)i;
        a[i] = sinf(f) * sinf(f);
        b[i] = cosf(f) * cosf(f);
    }
    gpu_warmup();
    cpu_warmup();
    for (int i = 1; i <= 10; ++i)
    {
        const size_t used_size = 50 * i * 1000 * 1000;
        printf("[test case %d]: data_size= %d B\n", i, used_size * sizeof(float));
        gpu_vadds(a, b, c1, used_size, 128, 1024);
        cpu_vadds(a, b, c2, used_size);
        verify(c1, c2, used_size);
    }

    delete a;
    delete b;
    delete c1;
    delete c2;
    return 0;
}