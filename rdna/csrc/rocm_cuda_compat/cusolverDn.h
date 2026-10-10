#pragma once
// Opaque stand-in. ATen/cuda/CUDAContextLight.h names cusolverDnHandle_t
// when CUDART_VERSION or USE_ROCM is set. This module does not call cuSOLVER.

typedef void* cusolverDnHandle_t;
