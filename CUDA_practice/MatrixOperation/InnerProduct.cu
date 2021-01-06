#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#define SIZE 33*1024
#define min(a, b) a > b ? b : a

const int threadperBlock = 256;
const int blocksperGrid = min(32, (SIZE + threadperBlock - 1) / threadperBlock);

__global__ void dot(float *a, float *b, float *c)
{
    __shared__ float cache[threadperBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < SIZE)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main()
{
    
    float *a, *b, *c_, c;
    float *d_a, *d_b, *d_c_;
    a = (float *)malloc(SIZE*sizeof(float));
    b = (float *)malloc(SIZE*sizeof(float));
    c_ = (float *)malloc(blocksperGrid*sizeof(float));

    cudaMalloc(&d_a, SIZE*sizeof(float));
    cudaMalloc(&d_b, SIZE*sizeof(float));
    cudaMalloc(&d_c_, blocksperGrid*sizeof(float));
    for (int i = 0; i < SIZE; ++i)
    {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(d_a, a, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    dot<<<blocksperGrid, threadperBlock>>>(d_a, d_b, d_c_);

    cudaMemcpy(c_, d_c_, blocksperGrid*sizeof(float), cudaMemcpyDeviceToHost);

    c = 0;
    for (int i = 0; i < blocksperGrid; i++)
    {
        c += c_[i];
    }
    printf("%.6g\n", c);
    
    float K = 0.;
    for (int i = 0; i < SIZE; i++)
    {
        K += i*i;
    }
    printf("%.6g\n", K);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_);
    
    free(a);
    free(b);
    free(c_);

    return 0;
}
