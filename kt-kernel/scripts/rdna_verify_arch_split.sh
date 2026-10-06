#!/usr/bin/env bash
# Confirm gfx1030 and gfx1100 device libraries are separate objects, that the
# gfx1030 code object disassembly contains no WMMA or MFMA, and that the WMMA
# translation unit is rejected for gfx1030. No GPU required.
# A library that merely linked is not a pass: the gfx1030 ISA has to be read.
# REQUIRE_DOT=1 also requires a packed DOT opcode (off by default).
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

rocm_tool() {
  local name="$1"
  local cand root rel
  if command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
    return 0
  fi
  local roots=()
  if [[ -n "${ROCM_PATH:-}" ]]; then
    roots+=("${ROCM_PATH}")
  fi
  roots+=(/opt/rocm)
  local rels=("bin/${name}" "libexec/${name}" "llvm/bin/${name}" "lib/llvm/bin/${name}")
  for root in "${roots[@]}"; do
    for rel in "${rels[@]}"; do
      cand="${root}/${rel}"
      if [[ -x "${cand}" ]]; then
        echo "${cand}"
        return 0
      fi
    done
  done
  return 1
}

# llvm-objdump prints one of these for the same packed-DOT opcode, depending
# on the encoding and the LLVM version. sdot2/sdot4/sdot8 lower to the i16,
# i8/iu8, and i4 forms (v_dot4c_i32_i8 on gfx10, v_dot4_i32_iu8 on gfx11).
DOT_RE='(^|[^[:alnum:]_])(v_dot2c?_f32_f16|v_dot4c?_i32_i8|v_dot4_i32_iu8|v_dot2c?_i32_i16|v_dot8c?_i32_i4|v_sdot[0-9])'
FORBIDDEN_RE='(^|[^[:alnum:]_])v_(wmma|mfma)_'

have_gpu_isa() {
  local file="$1"
  [[ -s "${file}" ]] || return 1
  grep -q -i -E 'elf64-amdgpu|file format .*amdgcn' "${file}" || return 1
  grep -q 'Disassembly of section' "${file}" || return 1
  grep -qE '(^|[^[:alnum:]_])[sv]_[a-z0-9]+' "${file}"
}

is_amdgpu_elf() {
  local path="$1"
  python3 - "${path}" <<'PY'
import sys
data = open(sys.argv[1], "rb").read(20)
ok = (
    len(data) >= 20
    and data[:4] == b"\x7fELF"
    and data[4] == 2
    and data[5] == 1
    and int.from_bytes(data[18:20], "little") == 224
)
sys.exit(0 if ok else 1)
PY
}

code_object_mcpu() {
  local co="$1"
  local triple
  triple="$(strings "${co}" | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9]+[a-z0-9]*' | head -n 1 || true)"
  if [[ -n "${triple}" ]]; then
    echo "${triple##*--}"
  else
    echo "gfx1030"
  fi
}

disassemble_elf() {
  local objdump="$1"
  local co="$2"
  local dest="$3"
  local mcpu
  if ! is_amdgpu_elf "${co}"; then
    return 0
  fi
  mcpu="$(code_object_mcpu "${co}")"
  "${objdump}" -d --triple=amdgcn --mcpu="${mcpu}" "${co}" >> "${dest}" 2>>"${dest}.err" || \
    "${objdump}" -d --mcpu="${mcpu}" "${co}" >> "${dest}" 2>>"${dest}.err" || \
    "${objdump}" -d --arch-name=amdgcn --mcpu="${mcpu}" "${co}" >> "${dest}" 2>>"${dest}.err" || true
}

disassemble_new_code_objects() {
  local objdump="$1"
  local work="$2"
  local dest="$3"
  local co
  while IFS= read -r -d '' co; do
    if [[ "${co}" == "${work}/gfx1030.so" ]]; then
      continue
    fi
    disassemble_elf "${objdump}" "${co}" "${dest}"
  done < <(find "${work}" -type f -print0)
}

extract_amdgpu_elfs() {
  local blob="$1"
  local dest_dir="$2"
  python3 - "${blob}" "${dest_dir}" <<'PY'
import sys
from pathlib import Path

data = Path(sys.argv[1]).read_bytes()
dest = Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
magic = b"\x7fELF"
EM_AMDGPU = 224
found = 0
start = 0
while True:
    j = data.find(magic, start)
    if j < 0:
        break
    start = j + 4
    if j + 64 > len(data) or data[j + 4] != 2 or data[j + 5] != 1:
        continue
    if int.from_bytes(data[j + 18 : j + 20], "little") != EM_AMDGPU:
        continue
    e_shoff = int.from_bytes(data[j + 40 : j + 48], "little")
    e_shentsize = int.from_bytes(data[j + 58 : j + 60], "little")
    e_shnum = int.from_bytes(data[j + 60 : j + 62], "little")
    if e_shentsize < 64 or e_shnum == 0 or e_shoff < 64:
        continue
    end = e_shoff + e_shentsize * e_shnum
    if j + end > len(data):
        continue
    ok = True
    for index in range(e_shnum):
        off = j + e_shoff + index * e_shentsize
        if off + 64 > len(data):
            ok = False
            break
        sh_type = int.from_bytes(data[off + 4 : off + 8], "little")
        sh_offset = int.from_bytes(data[off + 24 : off + 32], "little")
        sh_size = int.from_bytes(data[off + 32 : off + 40], "little")
        # SHT_NOBITS occupies no file bytes.
        if sh_type != 8:
            end = max(end, sh_offset + sh_size)
    if not ok or end < 64 or j + end > len(data):
        continue
    (dest / f"scan-{found}.co").write_bytes(data[j : j + end])
    found += 1
print(found)
PY
}

disassemble_gfx1030_so() {
  local so="$1"
  local dest="$2"
  local work="$3"
  local objdump="" rocobj="" readobj="" objcopy=""
  mkdir -p "${work}/extract" "${work}/roc"
  cp -a "${so}" "${work}/gfx1030.so"

  objdump="$(rocm_tool llvm-objdump || true)"
  rocobj="$(rocm_tool roc-obj || true)"
  readobj="$(rocm_tool llvm-readobj || true)"
  objcopy="$(rocm_tool llvm-objcopy || true)"

  if [[ -z "${objdump}" && -z "${rocobj}" ]]; then
    echo "llvm-objdump and roc-obj are both missing; cannot disassemble ${so}" >&2
    return 1
  fi

  if [[ -n "${rocobj}" ]]; then
    "${rocobj}" -d -t 'gfx1030' -o "${work}/roc" "${work}/gfx1030.so" \
      >"${work}/roc-obj.out" 2>"${work}/roc-obj.err" || true
    local asm
    shopt -s nullglob
    for asm in "${work}/roc"/*.s "${work}/roc"/*; do
      [[ -f "${asm}" ]] || continue
      if grep -q -i 'elf64-amdgpu' "${asm}" 2>/dev/null; then
        cat "${asm}" >> "${dest}"
      fi
    done
    shopt -u nullglob
  fi

  if [[ -n "${objdump}" ]] && ! have_gpu_isa "${dest}"; then
    "${objdump}" --offloading -d --triple=amdgcn --mcpu=gfx1030 "${work}/gfx1030.so" \
      >"${work}/offloading.txt" 2>"${work}/offloading.err" || true
    if grep -q -i -E 'elf64-amdgpu|file format .*amdgcn' "${work}/offloading.txt" \
      && grep -q 'Disassembly of section' "${work}/offloading.txt"; then
      cat "${work}/offloading.txt" >> "${dest}"
    fi
    if ! have_gpu_isa "${dest}"; then
      (
        cd "${work}/extract"
        "${objdump}" --offloading "${work}/gfx1030.so"
        "${objdump}" --offload-fatbin --arch-name='amdgcn-amd-amdhsa--gfx1030' "${work}/gfx1030.so"
      ) >"${work}/fatbin.txt" 2>"${work}/fatbin.err" || true
      disassemble_new_code_objects "${objdump}" "${work}" "${dest}"
    fi
  fi

  if [[ -n "${objdump}" && -n "${readobj}" && -n "${objcopy}" ]] && ! have_gpu_isa "${dest}"; then
    "${readobj}" --offloading "${work}/gfx1030.so" >"${work}/uris.txt" 2>"${work}/uris.err" || true
    local uri n=0
    while IFS= read -r uri; do
      [[ -n "${uri}" ]] || continue
      n=$((n + 1))
      (
        cd "${work}/extract"
        "${objcopy}" "--dump-offload-bundle=${uri}" "${work}/gfx1030.so"
      ) >"${work}/objcopy-${n}.out" 2>"${work}/objcopy-${n}.err" || true
    done < <(grep -oE '[^[:space:]"'\'']*offset=[0-9]+&size=[0-9]+' "${work}/uris.txt" || true)
    disassemble_new_code_objects "${objdump}" "${work}" "${dest}"
  fi

  if ! have_gpu_isa "${dest}" && command -v python3 >/dev/null 2>&1; then
    extract_amdgpu_elfs "${work}/gfx1030.so" "${work}/extract" >"${work}/scan.count" 2>"${work}/scan.err" || true
    if [[ -n "${objdump}" ]]; then
      disassemble_new_code_objects "${objdump}" "${work}" "${dest}"
    fi
  fi

  if ! have_gpu_isa "${dest}"; then
    echo "could not disassemble a gfx1030 code object inside ${so}" >&2
    echo "a successful build is not an arch-split pass" >&2
    if [[ -n "${objdump}" ]]; then
      echo "llvm-objdump: ${objdump}" >&2
    fi
    if [[ -n "${rocobj}" ]]; then
      echo "roc-obj: ${rocobj}" >&2
    fi
    local log
    for log in "${work}/offloading.err" "${work}/fatbin.err" "${work}/roc-obj.err" "${work}/uris.err" "${work}/scan.err" "${dest}.err"; do
      if [[ -s "${log}" ]]; then
        echo "---- ${log} ----" >&2
        tail -n 30 "${log}" >&2
      fi
    done
    echo "---- extracted files ----" >&2
    find "${work}" -type f -printf '%p %s\n' >&2 || true
    if [[ -s "${work}/uris.txt" ]]; then
      echo "---- offload listing ----" >&2
      head -n 40 "${work}/uris.txt" >&2
    fi
    if [[ -s "${work}/offloading.txt" ]]; then
      echo "---- offloading stdout ----" >&2
      head -n 40 "${work}/offloading.txt" >&2
    fi
    return 1
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

# Debian/Ubuntu strings scans only allocated sections unless --all is set.
# The marker can also sit in the embedded offload bundle.
so_strings() {
  strings -a "$1"
}

if so_strings "${GFX1030}" | grep -q 'KT_RDNA_WMMA_TU='; then
  echo "gfx1030 library contains the WMMA translation-unit marker" >&2
  exit 1
fi
if ! so_strings "${GFX1030}" | grep -q 'KT_RDNA_OBJECT_ARCH=gfx1030'; then
  echo "gfx1030 library is missing KT_RDNA_OBJECT_ARCH=gfx1030" >&2
  so_strings "${GFX1030}" | grep -F 'KT_RDNA' >&2 || true
  nm -D "${GFX1030}" | grep -F 'kt_rdna' >&2 || true
  exit 1
fi
if ! so_strings "${GFX1100}" | grep -q 'KT_RDNA_OBJECT_ARCH=gfx1100'; then
  echo "gfx1100 library is missing KT_RDNA_OBJECT_ARCH=gfx1100" >&2
  exit 1
fi
if ! so_strings "${WMMA}" | grep -q 'KT_RDNA_WMMA_TU=gfx1100'; then
  echo "WMMA library is missing KT_RDNA_WMMA_TU=gfx1100" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
DISASM="${TMP}/gfx1030.s"
: >"${DISASM}"
disassemble_gfx1030_so "${GFX1030}" "${DISASM}" "${TMP}/isa"

if grep -nE "${FORBIDDEN_RE}" "${DISASM}" >"${TMP}/forbidden"; then
  echo "gfx1030 code object contains v_wmma_* or v_mfma_* instructions:" >&2
  cat "${TMP}/forbidden" >&2
  exit 1
fi
echo "gfx1030 ISA: no v_wmma_* or v_mfma_*"

if [[ "${REQUIRE_DOT:-0}" == "1" ]]; then
  if ! grep -qE "${DOT_RE}" "${DISASM}"; then
    echo "REQUIRE_DOT=1 but gfx1030 disassembly has no packed DOT instruction" >&2
    echo "accepted: v_dot2_f32_f16 or v_dot2c_f32_f16, v_dot4_i32_i8 or v_dot4c_i32_i8, sdot forms (v_dot4_i32_iu8, v_dot2*_i32_i16, v_dot8*_i32_i4)" >&2
    exit 1
  fi
  echo "gfx1030 ISA: packed DOT present"
fi

HIPCC="${HIPCC:-hipcc}"
if ! command -v "${HIPCC}" >/dev/null 2>&1; then
  echo "hipcc not on PATH; ISA check passed, skipped the negative compile" >&2
  echo "arch split ok"
  exit 0
fi

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
