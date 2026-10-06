#!/usr/bin/env bash
# Confirm gfx1030 and gfx1100 device libraries are separate objects and that
# the WMMA translation unit is rejected for gfx1030. No GPU required.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${1:-}"
if [[ -z "${BUILD_DIR}" ]]; then
  echo "usage: $0 <directory-containing-libkt_rdna_*.so>" >&2
  exit 2
fi

need() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "missing ${path}" >&2
    exit 1
  fi
}

GFX1030="$(find "${BUILD_DIR}" -name 'libkt_rdna_gfx1030.so' -print -quit)"
GFX1100="$(find "${BUILD_DIR}" -name 'libkt_rdna_gfx1100.so' -print -quit)"
WMMA="$(find "${BUILD_DIR}" -name 'libkt_rdna_wmma_gfx1100.so' -print -quit)"
need "${GFX1030}"
need "${GFX1100}"
need "${WMMA}"

echo "gfx1030: ${GFX1030}"
echo "gfx1100: ${GFX1100}"
echo "wmma:    ${WMMA}"

if nm "${GFX1030}" | grep -q 'kt_rdna_wmma_entry'; then
  echo "gfx1030 library exports kt_rdna_wmma_entry" >&2
  exit 1
fi
if nm "${GFX1100}" | grep -q 'kt_rdna_wmma_entry'; then
  echo "non-WMMA gfx1100 library exports kt_rdna_wmma_entry" >&2
  exit 1
fi
if ! nm "${WMMA}" | grep -q 'kt_rdna_wmma_entry'; then
  echo "WMMA library does not export kt_rdna_wmma_entry" >&2
  exit 1
fi

if strings "${GFX1030}" | grep -q 'KT_RDNA_WMMA_TU='; then
  echo "gfx1030 library contains the WMMA translation-unit marker" >&2
  exit 1
fi
if ! strings "${GFX1030}" | grep -q 'KT_RDNA_OBJECT_ARCH=gfx1030'; then
  echo "gfx1030 library is missing KT_RDNA_OBJECT_ARCH=gfx1030" >&2
  exit 1
fi
if ! strings "${GFX1100}" | grep -q 'KT_RDNA_OBJECT_ARCH=gfx1100'; then
  echo "gfx1100 library is missing KT_RDNA_OBJECT_ARCH=gfx1100" >&2
  exit 1
fi
if ! strings "${WMMA}" | grep -q 'KT_RDNA_WMMA_TU=gfx1100'; then
  echo "WMMA library is missing KT_RDNA_WMMA_TU=gfx1100" >&2
  exit 1
fi

HIPCC="${HIPCC:-hipcc}"
if ! command -v "${HIPCC}" >/dev/null 2>&1; then
  echo "hipcc not on PATH; symbol checks passed, skipped the negative compile" >&2
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
set +e
NEG="$("${HIPCC}" --offload-arch=gfx1030 -c "${ROOT}/kt-kernel/rocm/wmma_gfx11.hip" \
  -DKT_RDNA_ARCH_TAG=gfx1030 -o "${TMP}/wmma_gfx1030.o" 2>&1)"
NEG_STATUS=$?
MIXED="$("${HIPCC}" --offload-arch=gfx1030 --offload-arch=gfx1100 -c \
  "${ROOT}/kt-kernel/rocm/wmma_gfx11.hip" -DKT_RDNA_ARCH_TAG=gfx1100 \
  -o "${TMP}/wmma_mixed.o" 2>&1)"
MIXED_STATUS=$?
set -e
if [[ "${NEG_STATUS}" -eq 0 ]]; then
  echo "WMMA source compiled for gfx1030; that object must not exist" >&2
  exit 1
fi
if ! grep -q 'non-gfx11' <<<"${NEG}"; then
  echo "gfx1030 WMMA compile failed for an unexpected reason:" >&2
  echo "${NEG}" >&2
  exit 1
fi
if [[ "${MIXED_STATUS}" -eq 0 ]]; then
  echo "mixed gfx1030+gfx1100 WMMA compile succeeded" >&2
  exit 1
fi
if ! grep -q 'non-gfx11' <<<"${MIXED}"; then
  echo "mixed WMMA compile failed for an unexpected reason:" >&2
  echo "${MIXED}" >&2
  exit 1
fi

echo "arch split ok"
