//===--- ConvertCherryToSCF.cpp -------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::cherry {

#define GEN_PASS_DEF_CONVERTCHERRYTOSCF
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

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
    rewriter.inlineRegionBefore(ifOp.getThenRegion(),
                                &scfIfOp.getThenRegion().back());
    rewriter.eraseBlock(&scfIfOp.getThenRegion().back());

    rewriter.inlineRegionBefore(ifOp.getElseRegion(),
                                &scfIfOp.getElseRegion().back());
    rewriter.eraseBlock(&scfIfOp.getElseRegion().back());

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

struct ConvertCherryToSCF
    : public impl::ConvertCherryToSCFBase<ConvertCherryToSCF> {

  using impl::ConvertCherryToSCFBase<
      ConvertCherryToSCF>::ConvertCherryToSCFBase;

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<cherry::CherryDialect, scf::SCFDialect>();
    target.addIllegalOp<cherry::IfOp>();
    target.addIllegalOp<cherry::YieldIfOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<IfOpLowering, YieldIfOpLowering>(&getContext());

    auto f = getOperation();
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(f, target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::cherry
