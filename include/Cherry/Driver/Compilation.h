#ifndef CHERRY_COMPILATION_H
#define CHERRY_COMPILATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

namespace cherry {

class Module;
class ParseResult;

class Compilation {
public:
  static auto make(std::string filename) -> std::unique_ptr<Compilation>;

  auto dumpTokens() -> int;
  auto dumpAST() -> int;
  auto dumpMLIR() -> int;

  auto sourceManager() -> llvm::SourceMgr& { return _sourceManager; };

private:
  llvm::SourceMgr _sourceManager;

  auto parse(std::unique_ptr<Module>& module) -> ParseResult;
};

}

#endif