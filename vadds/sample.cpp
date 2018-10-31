#include <math.h>
#include <stdio.h>
#include <omp.h>

#include "interfaces.h"

int main()
{
    const size_t arr_size = 512 * 1024 * 1024;
    float *a = new float[arr_size];
    float *b = new float[arr_size];
    float *c = new float[arr_size];

#pragma omp parallel for
    for (size_t i = 0; i < arr_size; ++i)
    {
        float f = (float)i;
        a[i] = sinf(f) * sinf(f);
        b[i] = cosf(f) * cosf(f);
    }
    
    gpu_vadds(a, b, c, arr_size, 128, 1024);
    verify_vadds(a, b, c, arr_size);
    return 0;
}