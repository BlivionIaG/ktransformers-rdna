"""HIP graph-capture rules ported from vllm-rdna ``rdna2_graph_keepalive``.

gfx1030 reports ``hipStreamIsCapturing`` during eager mixed prefill and
during replay, so the kernels do not trust that query. Callers (SGLang's
CUDA/HIP graph runner) drive the two flags instead:

1. ``set_capturing(True)`` immediately before beginning a FULL graph capture,
   and ``set_capturing(False)`` if capture is aborted.
2. ``freeze_capture_persist()`` after a successful FULL capture. That freezes
   the capture slots. Later eager prefill grows a different buffer, so a
   16k prefill cannot recycle a pointer baked into the graph.
3. Persist allocations are immortal. Do not free them and do not expect the
   PyTorch caching allocator to reuse those addresses.

``q_gemm`` outputs alias those persist buffers. During capture the Python
wrappers return the alias (stable address). Outside capture they clone.
"""

from __future__ import annotations

_capturing = False


def is_capturing() -> bool:
    return _capturing


def set_capturing(on: bool) -> None:
    global _capturing
    _capturing = bool(on)
    from ._ops import available, load

    if available():
        load().rdna2_set_graph_capturing(bool(on))


def freeze_capture_persist() -> None:
    global _capturing
    _capturing = False
    from ._ops import available, load

    if available():
        load().rdna2_freeze_capture_persist()
