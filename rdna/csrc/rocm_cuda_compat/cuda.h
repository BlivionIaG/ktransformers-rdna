#pragma once
// Driver-API name. PyTorch headers include <cuda.h> for CUDA_VERSION and
// a handful of error types. This module does not call the CUDA driver.

#include <cuda_runtime_api.h>

#ifndef CUDA_VERSION
#define CUDA_VERSION CUDART_VERSION
#endif

typedef hipError_t CUresult;
#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS hipSuccess
#endif
