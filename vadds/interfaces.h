#ifndef __ad93IFM09mf__
#define __ad93IFM09mf__

#include <stdlib.h>

void gpu_vadds(
    float *a, float *b, float *c, size_t arr_size,
    size_t grid_x, size_t block_x);
void verify_vadds( float *a, float *b, float *c, size_t arr_size);

#endif