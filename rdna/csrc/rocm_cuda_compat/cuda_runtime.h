#pragma once
// Host runtime umbrella. ROCm 6.4 does not ship this CUDA name.
// cuda_runtime_api.h in this directory supplies the aliases PyTorch uses.

#include <cuda_runtime_api.h>
#include <hip/hip_runtime.h>
