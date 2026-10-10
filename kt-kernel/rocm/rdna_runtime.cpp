#include "rdna_arch.h"

#include <dlfcn.h>

#include <mutex>
#include <string>

namespace kt::rdna {
namespace {
std::mutex g_mu;
void* g_wmma = nullptr;
}  // namespace

const char* compiled_archs() {
#if defined(KT_RDNA_COMPILED_ARCHS)
  return KT_RDNA_COMPILED_ARCHS;
#else
  return "";
#endif
}

std::string load_wmma_library(const std::string& gcn_arch, const std::string& library_path) {
  const std::string rejected = wmma_rejection(gcn_arch);
  if (!rejected.empty()) return rejected;
  if (library_path.empty()) return "WMMA library path is empty";

  std::lock_guard<std::mutex> lock(g_mu);
  if (g_wmma != nullptr) return {};

  void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* err = dlerror();
    return std::string("dlopen WMMA library failed: ") + (err != nullptr ? err : library_path);
  }
  dlerror();
  void* entry = dlsym(handle, "kt_rdna_wmma_entry");
  const char* sym_err = dlerror();
  if (entry == nullptr || sym_err != nullptr) {
    dlclose(handle);
    return "WMMA library is missing kt_rdna_wmma_entry (" + library_path + ")";
  }
  g_wmma = handle;
  return {};
}

}  // namespace kt::rdna
