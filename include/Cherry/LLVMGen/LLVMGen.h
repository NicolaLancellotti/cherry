//===--- LLVMGen.h - LLVM Generator -----------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_LLVMGEN_H
#define CHERRY_LLVMGEN_H

#include <memory>

namespace llvm {
class SourceMgr;
class Module;
class LLVMContext;
} // end namespace llvm

namespace cherry {
class Module;
class CherryResult;

auto llvmGen(const llvm::SourceMgr &sourceManager,
             llvm::LLVMContext &context,
             const Module &moduleAST,
             std::unique_ptr<llvm::Module> &module,
             bool enableOpt) -> CherryResult;

} // end namespace cherry

#endif // CHERRY_LLVMGEN_H
