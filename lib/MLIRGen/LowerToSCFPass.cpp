//===--- LowerToLLVMPass.cpp - Lowering to the llvm dialect ---------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "StructType.h"
#include "cherry/MLIRGen/CherryDialect.h"
#include "cherry/MLIRGen/CherryOps.h"
#include "cherry/MLIRGen/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct IfOpLowering : public ConversionPattern {
  IfOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::IfOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    Location loc = op->getLoc();
    Value operand = operands.front();
    auto ifOp = dyn_cast<cherry::IfOp>(op);

    auto scfIfOp = rewriter.create<mlir::scf::IfOp>(
        loc, ifOp.getResult().getType(), operand, true);
    rewriter.inlineRegionBefore(ifOp.thenRegion(),
                                &scfIfOp.thenRegion().back());
    rewriter.eraseBlock(&scfIfOp.thenRegion().back());

    rewriter.inlineRegionBefore(ifOp.elseRegion(),
                                &scfIfOp.elseRegion().back());
    rewriter.eraseBlock(&scfIfOp.elseRegion().back());

    rewriter.replaceOp(op, scfIfOp.getResult(0));
    return success();
  }
};

struct YieldIfOpLowering : public ConversionPattern {
  YieldIfOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::YieldIfOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    Value operand = operands.front();
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, operand);
    return success();
  }
};

struct CherryToSCFLoweringPass
    : public PassWrapper<CherryToSCFLoweringPass, OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<cherry::CherryDialect, scf::SCFDialect>();
    target.addIllegalOp<cherry::IfOp>();
    target.addIllegalOp<cherry::YieldIfOp>();

    OwningRewritePatternList patterns;
    patterns.insert<IfOpLowering>(&getContext());
    patterns.insert<YieldIfOpLowering>(&getContext());

    auto f = getOperation();
    if (failed(applyPartialConversion(f, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // end namespace

namespace mlir {

auto mlir::cherry::createLowerToSCFPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<CherryToSCFLoweringPass>();
}
auto registerLowerToSCFPass() -> void {
  PassRegistration<CherryToSCFLoweringPass>(
      "convert-cherry-to-scf", "Convert some Cherry ops to SCF ops");
}
} // namespace mlir
