#ifndef CHERRY_COMPILATION_H
#define CHERRY_COMPILATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

namespace cherry {

class Compilation {
public:
  static auto make(std::string filename) -> std::unique_ptr<Compilation>;

  auto dumpTokens() -> int;

  auto sourceManager() -> llvm::SourceMgr& { return _sourceManager; };

private:
  llvm::SourceMgr _sourceManager;
};

}

#endif