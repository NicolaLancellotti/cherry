//===--- Compilation.h - Compilation Task Data Structure --------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_COMPILATION_H
#define CHERRY_COMPILATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

namespace cherry {

class Module;
class CherryResult;

class Compilation {
public:
  static auto make(std::string filename,
                   bool enableOpt) -> std::unique_ptr<Compilation>;

  auto dumpTokens() -> int;
  auto dumpAST() -> int;
  auto dumpMLIR() -> int;

  auto sourceManager() -> llvm::SourceMgr& { return _sourceManager; };

private:
  llvm::SourceMgr _sourceManager;
  bool _enableOpt;

  auto parse(std::unique_ptr<Module>& module) -> CherryResult;
};

} // end namespace cherry

#endif // CHERRY_COMPILATION_H
