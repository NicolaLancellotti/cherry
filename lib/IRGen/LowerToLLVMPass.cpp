//===--- LowerToLLVMPass.cpp - Lowering to the llvm dialect ---------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/IRGen/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct CherryToLLVMLoweringPass
    : public PassWrapper<CherryToLLVMLoweringPass, OperationPass<ModuleOp>> {

  auto runOnOperation() -> void final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    LLVMTypeConverter typeConverter(&getContext());
    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, patterns)))
      signalPassFailure();
  }
};

} // end namespace

auto mlir::cherry::createLowerToLLVMPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<CherryToLLVMLoweringPass>();
}
