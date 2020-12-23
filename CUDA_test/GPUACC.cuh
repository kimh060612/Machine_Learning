#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda.h"
#include <iostream>
#include <cufft.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus

extern "C" 
{
#endif
    class GPUACC 
    {
	public:
		GPUACC(void);
		int sum_cuda(int a, int b, int* c);
		virtual ~GPUACC(void);
    };
    
    GPUACC::GPUACC(void) {

    }
    
    GPUACC::~GPUACC(void) {
    
    }
#ifdef __cplusplus
}

#endif