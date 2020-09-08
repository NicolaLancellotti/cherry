#ifndef CHERRY_MLIRGEN_H
#define CHERRY_MLIRGEN_H

namespace mlir {
class MLIRContext;
class OwningModuleRef;
}

namespace llvm {
class SourceMgr;
}

namespace cherry {
class Module;

auto mlirGen(const llvm::SourceMgr &sourceManager,
             mlir::MLIRContext &context,
             const Module &moduleAST) -> mlir::OwningModuleRef;

}

#endif