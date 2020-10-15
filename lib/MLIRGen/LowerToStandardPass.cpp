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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct ConstantOpLowering : public OpRewritePattern<cherry::ConstantOp> {
  using OpRewritePattern<cherry::ConstantOp>::OpRewritePattern;

  auto matchAndRewrite(cherry::ConstantOp op,
                       PatternRewriter &rewriter) const -> LogicalResult final {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.valueAttr());
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
    if (operands.size() == 0) {
      rewriter.replaceOpWithNewOp<ReturnOp>(op, llvm::None);
    } else {
      Value operand = operands.front();
      rewriter.replaceOpWithNewOp<ReturnOp>(op,llvm::ArrayRef<Type>(),
                                            operand);
    }
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
    llvm::SmallVector<Type, 1> results;
    if (callOp.getResults().size() != 0)
      results.push_back(callOp.getResult(0).getType());

    rewriter.replaceOpWithNewOp<CallOp>(op, callOp.callee(), results, operands);
    return success();
  }
};

struct IfOpLowering : public ConversionPattern {
  IfOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::IfOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op,
                       ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter)
  const -> LogicalResult final {
    Location loc = op->getLoc();
    Value operand = operands.front();
    auto ifOp = dyn_cast<cherry::IfOp>(op);

    auto scfIfOp = rewriter.create<mlir::scf::IfOp>(loc, ifOp.getResult().getType(),  operand, true);
    rewriter.inlineRegionBefore(ifOp.thenRegion(), &scfIfOp.thenRegion().back());
    rewriter.eraseBlock(&scfIfOp.thenRegion().back());

    rewriter.inlineRegionBefore(ifOp.elseRegion(), &scfIfOp.elseRegion().back());
    rewriter.eraseBlock(&scfIfOp.elseRegion().back());

    rewriter.replaceOp(op, scfIfOp.getResult(0));
    return success();
  }
};

struct WhileOpLowering : public ConversionPattern {
  WhileOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::WhileOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op,
                       ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter)
  const -> LogicalResult final {
    Location loc = op->getLoc();
    auto whileOp = dyn_cast<cherry::WhileOp>(op);

    auto *initialBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *afterLoopBlock = rewriter.splitBlock(initialBlock, opPosition);

    auto &bodyRegion = whileOp.bodyRegion();
    auto *bodyBlock = &bodyRegion.front();

    // Emit condition block
    auto &conditionRegion = whileOp.conditionRegion();
    Block *conditionBlock = &conditionRegion.front();
    Operation *conditionTerminator = conditionRegion.back().getTerminator();
    ValueRange conditionTerminatorOperands = conditionTerminator->getOperands();
    rewriter.setInsertionPointToEnd(&conditionRegion.back());
    Value operand = conditionTerminatorOperands.front();
    rewriter.create<CondBranchOp>(loc, operand,
                                  bodyBlock, ArrayRef<Value>(),
                                  afterLoopBlock, ArrayRef<Value>());
    rewriter.eraseOp(conditionTerminator);
    rewriter.inlineRegionBefore(conditionRegion, afterLoopBlock);

    // Emit body block
    Operation *bodyTerminator = bodyRegion.back().getTerminator();
    rewriter.eraseOp(bodyTerminator);
    rewriter.setInsertionPointToEnd(&bodyRegion.back());
    rewriter.create<BranchOp>(loc, conditionBlock, llvm::None);
    rewriter.inlineRegionBefore(bodyRegion, afterLoopBlock);

    // Emit branch to condition block
    rewriter.setInsertionPointToEnd(initialBlock);
    rewriter.create<BranchOp>(loc, conditionBlock);

    rewriter.replaceOp(whileOp, afterLoopBlock->getArguments());
    return success();
  }
};

struct YieldOpLowering : public ConversionPattern {
  YieldOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::YieldOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op,
                       ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter)
  const -> LogicalResult final {
    Value operand = operands.front();
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, operand);
    return success();
  }
};

struct ArithmeticLogicOpLowering : public ConversionPattern {
  ArithmeticLogicOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::ArithmeticLogicOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op,
                       ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter)
  const -> LogicalResult final {
    auto resultTypes = op->getResultTypes();
    auto arithmeticLogicOp = dyn_cast<cherry::ArithmeticLogicOp>(op);
    auto oper = arithmeticLogicOp.op();
    if (oper == "+")
      rewriter.replaceOpWithNewOp<AddIOp>(op, resultTypes, operands);
    else if (oper == "-")
      rewriter.replaceOpWithNewOp<SubIOp>(op, resultTypes, operands);
    else if (oper == "*")
      rewriter.replaceOpWithNewOp<MulIOp>(op, resultTypes, operands);
    else if (oper == "/")
      rewriter.replaceOpWithNewOp<UnsignedDivIOp>(op, resultTypes, operands);
    else if (oper == "%")
      rewriter.replaceOpWithNewOp<UnsignedRemIOp>(op, resultTypes, operands);
    else if (oper == "and")
      rewriter.replaceOpWithNewOp<AndOp>(op, resultTypes, operands);
    else if (oper == "or")
      rewriter.replaceOpWithNewOp<OrOp>(op, resultTypes, operands);
    else if (oper == "eq")
      rewriter.replaceOpWithNewOp<CmpIOp>(op, resultTypes, mlir::CmpIPredicate::eq,
                                          operands[0], operands[1]);
    else if (oper == "neq")
      rewriter.replaceOpWithNewOp<CmpIOp>(op, resultTypes, mlir::CmpIPredicate::ne,
                                          operands[0], operands[1]);
    else if (oper == "lt")
      rewriter.replaceOpWithNewOp<CmpIOp>(op, resultTypes, mlir::CmpIPredicate::ult,
                                          operands[0], operands[1]);
    else if (oper == "le")
      rewriter.replaceOpWithNewOp<CmpIOp>(op, resultTypes, mlir::CmpIPredicate::ule,
                                          operands[0], operands[1]);
    else if (oper == "gt")
      rewriter.replaceOpWithNewOp<CmpIOp>(op, resultTypes, mlir::CmpIPredicate::ugt,
                                          operands[0], operands[1]);
    else if (oper == "ge")
      rewriter.replaceOpWithNewOp<CmpIOp>(op, resultTypes, mlir::CmpIPredicate::uge,
                                          operands[0], operands[1]);
    return success();
  }
};

struct CherryToStandardLoweringPass
    : public PassWrapper<CherryToStandardLoweringPass, FunctionPass> {

  auto runOnFunction() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addIllegalDialect<cherry::CherryDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalOp<cherry::PrintOp>();
    target.addLegalOp<cherry::CastOp>();

    OwningRewritePatternList patterns;
    patterns.insert<ReturnOpLowering>(&getContext());
    patterns.insert<ConstantOpLowering>(&getContext());
    patterns.insert<CallOpLowering>(&getContext());
    patterns.insert<YieldOpLowering>(&getContext());
    patterns.insert<IfOpLowering>(&getContext());
    patterns.insert<WhileOpLowering>(&getContext());
    patterns.insert<ArithmeticLogicOpLowering>(&getContext());

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
