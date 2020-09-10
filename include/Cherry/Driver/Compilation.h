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
#include "mlir/IR/MLIRContext.h"

namespace mlir {
class OwningModuleRef;
} // end namespace mlir

namespace cherry {

class Module;
class CherryResult;

class Compilation {
public:
  enum Lowering {
    None,
    Standard,
    LLVM
  };

  static auto make(std::string filename,
                   bool enableOpt) -> std::unique_ptr<Compilation>;

  auto dumpTokens() -> int;
  auto dumpAST() -> int;
  auto dumpMLIR(Lowering lowering) -> int;

  auto sourceManager() -> llvm::SourceMgr& { return _sourceManager; };

private:
  llvm::SourceMgr _sourceManager;
  bool _enableOpt;
  mlir::MLIRContext _context;

  auto parse(std::unique_ptr<Module>& module) -> CherryResult;
  auto genMLIR(mlir::OwningModuleRef& module,
               Lowering lowering) -> CherryResult;
};

} // end namespace cherry

#endif // CHERRY_COMPILATION_H
