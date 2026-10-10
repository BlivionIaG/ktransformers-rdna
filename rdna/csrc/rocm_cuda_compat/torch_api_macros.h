#pragma once
// This ROCm PyTorch wheel uses TORCH_CUDA_CPP_API in ATen/cuda headers but
// does not define it (the HIP export macro is TORCH_HIP_CPP_API). Empty is
// enough: these translation units only call into libtorch, they do not
// export ATen symbols.

#ifndef TORCH_CUDA_CPP_API
#define TORCH_CUDA_CPP_API
#endif
#ifndef TORCH_CUDA_CU_API
#define TORCH_CUDA_CU_API __host__ __device__
#endif
