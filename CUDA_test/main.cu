#include <iostream>
#include "GPUACC.cuh"

using namespace std;

int sum_int(int a, int b);
int sum_int(int a, int b) {
	int c;
	c = a + b;
	return c;
}


int main() 
{
	int a, b, c;
	int d;
	a = 3; 
	b = 4;
	c = sum_int(3, 4);
	GPUACC gpuacc;
	gpuacc.sum_cuda(a, b, &d);
	cout << "CPU를 통한 합 : " << a << "와 " << b << "의 합은 " << c << "입니다." << endl;
	cout << "GPU를 통한 합 : " << a << "와 " << b << "의 합은 " << d << "입니다." << endl;
	system("pause");
	return 0;

}


__global__ void sum_kernel(int a, int b, int *c) 
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	c[tid] = a + b;
}

int GPUACC::sum_cuda(int a, int b, int *c) 
{
	int* f;
	cudaMalloc((void**)&f, sizeof(int) * 1);
	cudaMemcpy(f, c, sizeof(int) * 1, cudaMemcpyHostToDevice);
	sum_kernel << <1, 1 >> > (a, b, f);
	cudaMemcpy(c, f, sizeof(int) * 1, cudaMemcpyDefault);
	cudaFree(f);

	return true;
}