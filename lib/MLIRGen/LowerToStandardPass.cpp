//===--- LowerToStandardPass.cpp - Lowering to the standard dialect -------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_LOWERTOSTANDARDPASS_H
#define CHERRY_LOWERTOSTANDARDPASS_H

#include "cherry/MLIRGen/CherryDialect.h"
#include "cherry/MLIRGen/CherryOps.h"
#include "cherry/MLIRGen/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

static Value insertAlloca(MemRefType type, Location loc,
                          PatternRewriter &rewriter) {
  auto alloca = rewriter.create<AllocaOp>(loc, type);

  auto *parentBlock = alloca.getOperation()->getBlock();
  alloca.getOperation()->moveBefore(&parentBlock->front());

//  auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
//  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloca;
}

struct ConstantOpLowering : public OpRewritePattern<cherry::ConstantOp> {
  using OpRewritePattern<cherry::ConstantOp>::OpRewritePattern;

  auto matchAndRewrite(cherry::ConstantOp op,
                       PatternRewriter &rewriter) const -> LogicalResult final {

    Location loc = op.getLoc();
    mlir::Type argType = op.getResult().getType();
    MemRefType memRefType = MemRefType::get({}, argType);
    auto alloca = insertAlloca(memRefType, loc, rewriter);

    auto c = rewriter.create<ConstantOp>(loc, op.valueAttr());
    rewriter.create<StoreOp>(loc, c, alloca);

    rewriter.replaceOp(op, alloca);
    return success();
  }
};

struct ReturnOpLowering : public ConversionPattern {
  ReturnOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::ReturnOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op,
                       ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter)
  const -> LogicalResult final {
    Value operand = rewriter.create<LoadOp>(op->getLoc(), operands.front());
    rewriter.replaceOpWithNewOp<ReturnOp>(op,
                                          llvm::ArrayRef<Type>(),
                                          operand);
    return success();
  }
};

struct CallOpLowering : public ConversionPattern {

  CallOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::CallOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op,
                       ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const -> LogicalResult final {
    auto callOp = dyn_cast<cherry::CallOp>(op);

    SmallVector<Value, 3> loads;
    for (auto operand : operands) {
      if (operand.getType().isInteger(64)) {
        loads.push_back(operand);
      } else {
        loads.push_back(rewriter.create<LoadOp>(op->getLoc(), operand));
      }
    }

    rewriter.replaceOpWithNewOp<CallOp>(op, callOp.callee(),
                                        callOp.getResult().getType(),
                                        loads);
    return success();
  }
};

struct CherryToStandardLoweringPass
    : public PassWrapper<CherryToStandardLoweringPass, FunctionPass> {

  auto runOnFunction() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addIllegalDialect<cherry::CherryDialect>();
    target.addLegalOp<cherry::PrintOp>();
    target.addLegalOp<cherry::CastOp>();

    OwningRewritePatternList patterns;
    patterns.insert<ReturnOpLowering>(&getContext());
    patterns.insert<ConstantOpLowering>(&getContext());
    patterns.insert<CallOpLowering>(&getContext());

    auto f = getFunction();
    if (failed(applyPartialConversion(f,target, patterns)))
      signalPassFailure();
  }

};

} // end namespace

namespace mlir {

auto mlir::cherry::createLowerToStandardPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<CherryToStandardLoweringPass>();
}

auto registerLowerToStandardPass() -> void {
  PassRegistration<CherryToStandardLoweringPass>("lower-cherry-to-std",
                                                 " Lower Cherry operations to a combination of Standard and Cherry operations");
}

}

#endif // CHERRY_LOWERTOSTANDARDPASS_H
