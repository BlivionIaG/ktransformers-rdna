#pragma once

#include <string>

// Host-side arch gate. WMMA device code lives in libkt_rdna_wmma_<arch>.so and
// is dlopen'd only after this returns empty. gfx1030 is never on the list.

namespace kt::rdna {

inline std::string normalize_gcn_arch(std::string arch) {
  const auto colon = arch.find(':');
  if (colon != std::string::npos) arch.resize(colon);
  const auto space = arch.find_first_of(" \t");
  if (space != std::string::npos) arch.resize(space);
  return arch;
}

inline bool is_wmma_arch(const std::string& gcn_arch) {
  const std::string arch = normalize_gcn_arch(gcn_arch);
  static const char* kAllowed[] = {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151"};
  for (const char* name : kAllowed) {
    if (arch == name) return true;
  }
  return false;
}

inline std::string wmma_rejection(const std::string& gcn_arch) {
  if (is_wmma_arch(gcn_arch)) return {};
  const std::string arch = normalize_gcn_arch(gcn_arch);
  return "refusing to load gfx1100 WMMA code on '" + arch +
         "' (allowed: gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151). "
         "A gfx1030 process must not map the WMMA library.";
}

const char* compiled_archs();
std::string load_wmma_library(const std::string& gcn_arch, const std::string& library_path);

}  // namespace kt::rdna
