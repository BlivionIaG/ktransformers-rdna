#pragma once
// Compile-only CUDA runtime names for ROCm 6.4+.
//
// hip-dev 6.4 dropped the cuda_runtime_api.h alias that PyTorch's
// ATen/cuda and c10/cuda headers still include. These macros exist so
// those headers and the ported kernels (cudaStream_t) can parse. They
// are not a CUDA runtime. Device code in this module calls HIP and
// amdgcn packed DOT directly.

#include <hip/hip_runtime_api.h>

#ifndef CUDART_VERSION
#define CUDART_VERSION 11070
#endif

#ifndef cudaError_t
#define cudaError_t hipError_t
#endif
#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif
#ifndef cudaErrorNotReady
#define cudaErrorNotReady hipErrorNotReady
#endif
#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#endif
#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif

#ifndef cudaStream_t
#define cudaStream_t hipStream_t
#endif
#ifndef cudaStreamQuery
#define cudaStreamQuery hipStreamQuery
#endif
#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif
#ifndef cudaStreamGetPriority
#define cudaStreamGetPriority hipStreamGetPriority
#endif
#ifndef cudaDeviceGetStreamPriorityRange
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#endif

#ifndef cudaDeviceProp
#define cudaDeviceProp hipDeviceProp_t
#endif

#ifndef cudaMemcpyKind
#define cudaMemcpyKind hipMemcpyKind
#endif
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync hipMemcpyAsync
#endif

#ifndef cudaEvent_t
#define cudaEvent_t hipEvent_t
#endif
#ifndef cudaEventDefault
#define cudaEventDefault hipEventDefault
#endif
#ifndef cudaEventDisableTiming
#define cudaEventDisableTiming hipEventDisableTiming
#endif
#ifndef cudaEventCreateWithFlags
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#endif
#ifndef cudaEventDestroy
#define cudaEventDestroy hipEventDestroy
#endif
#ifndef cudaEventRecord
#define cudaEventRecord hipEventRecord
#endif
#ifndef cudaEventQuery
#define cudaEventQuery hipEventQuery
#endif
#ifndef cudaEventSynchronize
#define cudaEventSynchronize hipEventSynchronize
#endif
#ifndef cudaEventElapsedTime
#define cudaEventElapsedTime hipEventElapsedTime
#endif
#ifndef cudaStreamWaitEvent
#define cudaStreamWaitEvent hipStreamWaitEvent
#endif
#ifndef cudaDeviceSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#endif

#ifndef cudaStreamCaptureMode
#define cudaStreamCaptureMode hipStreamCaptureMode
#endif
#ifndef cudaStreamCaptureStatus
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#endif
#ifndef cudaStreamCaptureStatusNone
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#endif
#ifndef cudaStreamCaptureStatusActive
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#endif
#ifndef cudaStreamCaptureStatusInvalidated
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated
#endif
#ifndef cudaThreadExchangeStreamCaptureMode
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#endif
#ifndef cudaStreamIsCapturing
#define cudaStreamIsCapturing hipStreamIsCapturing
#endif
