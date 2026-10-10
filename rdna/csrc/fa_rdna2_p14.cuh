              "key_cache H/D mismatch");
  TORCH_CHECK(value_cache.size(1) == H && value_cache.size(2) == D &&
                  value_cache.size(3) == block_size,
              "value_cache shape mismatch");
  if (num_tokens == 0) {
    return;
  }

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(key));
  auto stream = c10::cuda::getCurrentCUDAStream();
  dim3 grid(num_tokens);
  dim3 block(128);
  const half* k_ptr = reinterpret_cast<const half*>(key.data_ptr());
  const half* v_ptr = reinterpret_cast<const half*>(value.data_ptr());
  half* kc_ptr = reinterpret_cast<half*>(key_cache.data_ptr());
  half* vc_ptr = reinterpret_cast<half*>(value_cache.data_ptr());
  const int64_t ks0 = key.stride(0);
  const int64_t vs0 = value.stride(0);
  const int64_t k0 = key_cache.stride(0);
  const int64_t k1 = key_cache.stride(1);
  const int64_t k2 = key_cache.stride(2);
  const int64_t k3 = key_cache.stride(3);
  const int64_t k4 = key_cache.stride(4);
  const int64_t v0 = value_cache.stride(0);
  const int64_t v1 = value_cache.stride(1);
  const int64_t v2 = value_cache.stride(2);
  const int64_t v3 = value_cache.stride(3);
  if (slot_mapping.scalar_type() == torch::kInt) {
    reshape_and_cache_flash_rdna2_kernel<int32_t>
        <<<grid, block, 0, stream.stream()>>>(
            k_ptr, v_ptr, kc_ptr, vc_ptr, slot_mapping.data_ptr<int32_t>(),
            num_tokens, H, D, block_size, x, ks0, vs0, k0, k1, k2, k3, k4,
            v0, v1, v2, v3, key_cache.size(0));
  } else {
    reshape_and_cache_flash_rdna2_kernel<int64_t>
        <<<grid, block, 0, stream.stream()>>>(
            k_ptr, v_ptr, kc_ptr, vc_ptr, slot_mapping.data_ptr<int64_t>(),
            num_tokens, H, D, block_size, x, ks0, vs0, k0, k1, k2, k3, k4,
            v0, v1, v2, v3, key_cache.size(0));
  }
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess,
              "reshape_and_cache_flash_rdna2 launch failed: ",
              hipGetErrorString(err));
}

torch::Tensor rdna2_immortal_zeros_from_ref(torch::Tensor ref,
                                            at::IntArrayRef size) {
  TORCH_CHECK(ref.is_cuda(), "rdna2_immortal_zeros ref must be CUDA/HIP");
  auto opts = torch::TensorOptions().dtype(ref.dtype()).device(ref.device());
  return rdna2_immortal_zeros(size, opts);
}
