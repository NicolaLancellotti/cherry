//===--- cherry-opt.cpp - Cherry optimiser ----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "cherry/MLIRGen/CherryDialect.h"

namespace mlir {
auto registerLowerToStandardPass() -> void;
auto registerLowerToSCFPass() -> void;
auto registerLowerToLLVMPass() -> void;
} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerLowerToStandardPass();
  mlir::registerLowerToSCFPass();
  mlir::registerLowerToLLVMPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::cherry::CherryDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "Cherry optimizer driver\n", registry));
}
