//===--- Compilation.h - Compilation Task Data Structure --------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_COMPILATION_H
#define CHERRY_COMPILATION_H

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/LLVMContext.h"

namespace mlir {
class OwningModuleRef;
} // end namespace mlir

namespace llvm {
class Module;
} // end namespace llvm

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

  static auto make(llvm::StringRef filename,
                   bool enableOpt,
                   bool backendLLVM) -> std::unique_ptr<Compilation>;

  auto dumpTokens() -> int;
  auto dumpParse() -> int;
  auto dumpAST() -> int;
  auto dumpMLIR(Lowering lowering) -> int;
  auto dumpLLVM() -> int;

  auto typecheck() -> int;
  auto jit() -> int;
  auto genObjectFile(const char *outputFileName) -> int;

  auto sourceManager() -> llvm::SourceMgr& { return _sourceManager; };

private:
  llvm::SourceMgr _sourceManager;
  bool _enableOpt;
  bool _backendLLVM;
  mlir::MLIRContext _mlirContext;
  llvm::LLVMContext _llvmContext;

  auto parse(std::unique_ptr<Module> &module) -> CherryResult;
  auto genMLIR(mlir::OwningModuleRef &module,
               Lowering lowering) -> CherryResult;
  auto genLLVM(std::unique_ptr<llvm::Module> &llvmModule) -> CherryResult;
};

} // end namespace cherry

#endif // CHERRY_COMPILATION_H
