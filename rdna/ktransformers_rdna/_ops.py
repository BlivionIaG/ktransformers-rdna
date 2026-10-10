"""Lazy loader for the HIP extension."""

from __future__ import annotations

_C = None
_ERR: Exception | None = None


class RdnaExtensionNotBuilt(RuntimeError):
    pass


def available() -> bool:
    try:
        load()
    except RdnaExtensionNotBuilt:
        return False
    return True


def load():
    global _C, _ERR
    if _C is not None:
        return _C
    if _ERR is not None:
        raise RdnaExtensionNotBuilt(str(_ERR)) from _ERR
    try:
        from . import _C as ext  # type: ignore
    except ImportError as exc:
        _ERR = exc
        raise RdnaExtensionNotBuilt(
            "ktransformers_rdna._C is not built. From rdna/: "
            "PYTORCH_ROCM_ARCH=gfx1030;gfx1100 python setup.py build_ext --inplace"
        ) from exc
    _C = ext
    return ext
