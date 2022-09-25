//===--- MLIRGen.h - MLIR Generator -----------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_MLIRGEN_H
#define CHERRY_MLIRGEN_H

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // end namespace mlir

namespace llvm {
class SourceMgr;
} // end namespace llvm

namespace cherry {
class Module;
class CherryResult;

auto mlirGen(const llvm::SourceMgr &sourceManager, mlir::MLIRContext &context,
             const Module &moduleAST, mlir::OwningOpRef<mlir::ModuleOp> &module)
    -> CherryResult;

} // end namespace cherry

#endif // CHERRY_MLIRGEN_H
