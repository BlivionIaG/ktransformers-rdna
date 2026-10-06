// Torch extension entry for the gfx1030 packed-DOT kernels.
//
// Op names match the vllm-rdna `_rocm_C` ops so the Python wrappers can
// follow the same contracts. Graph-capture flags are the ones defined in
// rdna2_graph_keepalive.cuh (source: opengfx1030/vllm-rdna).

#include <atomic>
#include <optional>

#include <torch/extension.h>

#include "rdna2_graph_keepalive.cuh"

std::atomic<int> g_rdna2_graph_capturing{0};
std::atomic<int> g_rdna2_capture_frozen{0};

void rdna2_set_graph_capturing(bool on) {
  if (on) {
    g_rdna2_capture_frozen.store(0, std::memory_order_release);
  }
  g_rdna2_graph_capturing.store(on ? 1 : 0, std::memory_order_release);
}

void rdna2_freeze_capture_persist() {
  g_rdna2_capture_frozen.store(1, std::memory_order_release);
  g_rdna2_graph_capturing.store(0, std::memory_order_release);
}

torch::Tensor gptq_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor gptq_gemm_rdna2_prefill(torch::Tensor a, torch::Tensor b_q_weight,
                                      torch::Tensor b_qzeros,
                                      torch::Tensor b_scales,
                                      torch::Tensor b_g_idx,
                                      bool use_v2_format);

at::Tensor gemv_f16_rdna2(const at::Tensor& x, const at::Tensor& w,
                          const std::optional<at::Tensor>& bias);

void moe_gptq_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk, bool fp32_accum);

void fa_rdna2_decode_paged(torch::Tensor Q, torch::Tensor key_cache,
                           torch::Tensor value_cache, torch::Tensor block_table,
                           torch::Tensor seq_lens, int64_t block_size,
                           int64_t kv_splits, int64_t sliding_window,
                           double scale, torch::Tensor out,
                           c10::optional<torch::Tensor> cu_query_lens);

void fa_rdna2_prefill_paged_varlen(torch::Tensor Q, torch::Tensor key_cache,
                                   torch::Tensor value_cache,
                                   torch::Tensor block_table,
                                   torch::Tensor cu_query_lens,
                                   torch::Tensor seq_lens, int64_t block_size,
                                   int64_t causal, int64_t sliding_window,
                                   double scale, torch::Tensor out);

void fa_rdna2_prefill_paged_varlen_short(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t sliding_window, double scale, torch::Tensor out);

void fa_rdna2_prefill_paged_varlen_splitk(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t kv_splits, int64_t sliding_window, double scale, torch::Tensor out);

void fa_rdna2_prefill_paged_varlen_gqa(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t sliding_window, double scale, torch::Tensor out);

namespace {

c10::optional<torch::Tensor> to_c10(std::optional<torch::Tensor> t) {
  if (t.has_value()) {
    return c10::optional<torch::Tensor>(std::move(*t));
  }
  return c10::nullopt;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rdna2_set_graph_capturing", &rdna2_set_graph_capturing,
        "Mark the start or end of an explicit HIP graph capture. "
        "hipStreamIsCapturing is not reliable on gfx1030.");
  m.def("rdna2_freeze_capture_persist", &rdna2_freeze_capture_persist,
        "Freeze FULL-graph persist slots so later eager prefill cannot "
        "recycle them.");
  m.def("gptq_gemm_rdna2", &gptq_gemm_rdna2,
        "W4A16 GPTQ decode GEMM (packed DOT). Output aliases a persist buffer.");
  m.def("gptq_gemm_rdna2_prefill", &gptq_gemm_rdna2_prefill,
        "W4A16 GPTQ prefill GEMM (packed DOT). Output aliases a persist buffer.");
  m.def("gemv_f16_rdna2", &gemv_f16_rdna2,
        "fp16 skinny GEMV, M in 1..8: y = x @ w.T (+ bias).");
  m.def("moe_gptq_gemm_rdna2", &moe_gptq_gemm_rdna2,
        "GPTQ-Int4 fused expert GEMM for resident experts.");
  m.def(
      "fa_rdna2_decode_paged",
      [](torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
         torch::Tensor block_table, torch::Tensor seq_lens, int64_t block_size,
         int64_t kv_splits, int64_t sliding_window, double scale,
         torch::Tensor out, std::optional<torch::Tensor> cu_query_lens) {
        fa_rdna2_decode_paged(Q, key_cache, value_cache, block_table, seq_lens,
                              block_size, kv_splits, sliding_window, scale, out,
                              to_c10(std::move(cu_query_lens)));
      },
      "GQA paged decode, head dim 128 or 256. Writes `out`.");
  m.def("fa_rdna2_prefill_paged_varlen", &fa_rdna2_prefill_paged_varlen,
        "GQA paged varlen prefill. Writes `out`.");
  m.def("fa_rdna2_prefill_paged_varlen_short",
        &fa_rdna2_prefill_paged_varlen_short,
        "Head-dim-128 paged varlen prefill for KV under 4096. Writes `out`.");
  m.def("fa_rdna2_prefill_paged_varlen_splitk",
        &fa_rdna2_prefill_paged_varlen_splitk,
        "Paged varlen prefill with split-K over the KV range. Writes `out`.");
  m.def("fa_rdna2_prefill_paged_varlen_gqa", &fa_rdna2_prefill_paged_varlen_gqa,
        "GQA paged varlen prefill with the softmax in registers. Writes `out`.");
}
