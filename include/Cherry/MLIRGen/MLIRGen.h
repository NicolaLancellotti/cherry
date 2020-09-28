//===--- MLIRGen.h - MLIR Generator -----------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_MLIRGEN_H
#define CHERRY_MLIRGEN_H

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // end namespace mlir

namespace llvm {
class SourceMgr;
} // end namespace llvm

namespace cherry {
class Module;
class CherryResult;

auto mlirGen(const llvm::SourceMgr &sourceManager,
             mlir::MLIRContext &context,
             const Module &moduleAST,
             mlir::OwningModuleRef &module) -> CherryResult;

} // end namespace cherry

#endif // CHERRY_MLIRGEN_H
