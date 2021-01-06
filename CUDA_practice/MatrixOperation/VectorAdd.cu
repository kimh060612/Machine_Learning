#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#define SIZE 1024

// __global__을 통해서 커널임을 표시한다. host에서 호출된다.
__global__ void VectorAdd(int *a, int *b, int *c, int n)
{
// 수많은 스레드가 동시에 처리한다.
// 따라서 threadIdx(스레드 인덱스)를 통해서 스레드들을 구별한다.
    int i = threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

__global__ void VectorBlockAdd(int *a, int *b, int *c, int n)
{
    int tid = blockIdx.x;
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}

int main()
{
    int *a, *b, *c, *d;
    int *d_a, *d_b, *d_c, *d_d;
    // 호스트의 메모리에 할당한다.
    a = (int *)malloc(SIZE*sizeof(int));
    b = (int *)malloc(SIZE*sizeof(int));
    c = (int *)malloc(SIZE*sizeof(int));
    d = (int *)malloc(SIZE*sizeof(int));
    // cudaMalloc(destination, number of byte)로 device의 메모리를 할당한다.
    cudaMalloc(&d_a, SIZE*sizeof(int));
    cudaMalloc(&d_b, SIZE*sizeof(int));
    cudaMalloc(&d_c, SIZE*sizeof(int));
    cudaMalloc(&d_d, SIZE*sizeof(int));
    // 초기화
    for (int i = 0; i<SIZE; ++i)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
        d[i] = 0;
    }
    // cudaMemcpy(destination, source, number of byte, cudaMemcpyHostToDevice)로 호스트에서 디바이스로 메모리를 카피한다.
    cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    // 함수 호출을 위해서 새로운 신텍스 요소를 추가할 필요가 있다.
    // 첫번째 parameter는 블럭의 수이다. 예제에서는 스레드 블럭이 하나이다.
    // SIZE는 1024개의 스레드를 의미한다.
    VectorAdd << < 1, SIZE >> >(d_a, d_b, d_c, SIZE);
    VectorBlockAdd << < SIZE, 1 >> >(d_a, d_b, d_d, SIZE);
    //cudaMemcpy(source, destination, number of byte, cudaMemDeviceToHost)로 디바이스의 메모리(연산 결과 데이터)를 호스트에 카피한다.
    cudaMemcpy(a, d_a, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d, d_d, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    printf("vector c:  \n");
    for (int i = 0; i<SIZE; ++i)
        printf("c[%d] = %d\n", i, c[i]);
    printf("vector d:  \n");
    for (int i = 0; i<SIZE; ++i)
        printf("d[%d] = %d\n", i, d[i]);
    // 호스트의 메모리 할당 해제
    free(a);
    free(b);
    free(c);
    free(d);
    // cudaFree(d_a)를 통해 디바이스의 메모리를 할당 해제
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    return 0;
}

