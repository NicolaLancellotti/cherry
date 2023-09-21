//===--- ConvertCherryToArithCfFunc.cpp -----------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::cherry {

#define GEN_PASS_DEF_CONVERTCHERRYTOARITHCFFUNC
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// Arith
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<cherry::ConstantOp> {
  using OpRewritePattern<cherry::ConstantOp>::OpRewritePattern;

  auto matchAndRewrite(cherry::ConstantOp op, PatternRewriter &rewriter) const
      -> LogicalResult final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValueAttr());
    return success();
  }
};

struct ArithmeticLogicOpLowering : public ConversionPattern {
  ArithmeticLogicOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::ArithmeticLogicOp::getOperationName(), 1,
                          ctx) {}

  auto matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    auto resultTypes = op->getResultTypes();
    auto arithmeticLogicOp = dyn_cast<cherry::ArithmeticLogicOp>(op);
    auto oper = arithmeticLogicOp.getOp();
    if (oper == "+")
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, resultTypes, operands);
    else if (oper == "-")
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, resultTypes, operands);
    else if (oper == "*")
      rewriter.replaceOpWithNewOp<arith::MulIOp>(op, resultTypes, operands);
    else if (oper == "/")
      rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, resultTypes, operands);
    else if (oper == "%")
      rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, resultTypes, operands);
    else if (oper == "and")
      rewriter.replaceOpWithNewOp<arith::AndIOp>(op, resultTypes, operands);
    else if (oper == "or")
      rewriter.replaceOpWithNewOp<arith::OrIOp>(op, resultTypes, operands);
    else if (oper == "eq")
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          op, resultTypes, arith::CmpIPredicate::eq, operands[0], operands[1]);
    else if (oper == "neq")
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          op, resultTypes, arith::CmpIPredicate::ne, operands[0], operands[1]);
    else if (oper == "lt")
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          op, resultTypes, arith::CmpIPredicate::ult, operands[0], operands[1]);
    else if (oper == "le")
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          op, resultTypes, arith::CmpIPredicate::ule, operands[0], operands[1]);
    else if (oper == "gt")
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          op, resultTypes, arith::CmpIPredicate::ugt, operands[0], operands[1]);
    else if (oper == "ge")
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          op, resultTypes, arith::CmpIPredicate::uge, operands[0], operands[1]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Cf
//===----------------------------------------------------------------------===//

struct WhileOpLowering : public ConversionPattern {
  WhileOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::WhileOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    Location loc = op->getLoc();
    auto whileOp = dyn_cast<cherry::WhileOp>(op);

    auto *initialBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *afterLoopBlock = rewriter.splitBlock(initialBlock, opPosition);

    auto &bodyRegion = whileOp.getBodyRegion();
    auto *bodyBlock = &bodyRegion.front();

    // Emit condition block
    auto &conditionRegion = whileOp.getConditionRegion();
    Block *conditionBlock = &conditionRegion.front();
    Operation *conditionTerminator = conditionRegion.back().getTerminator();
    ValueRange conditionTerminatorOperands = conditionTerminator->getOperands();
    rewriter.setInsertionPointToEnd(&conditionRegion.back());
    Value operand = conditionTerminatorOperands.front();
    rewriter.create<mlir::cf::CondBranchOp>(loc, operand, bodyBlock,
                                            ArrayRef<Value>(), afterLoopBlock,
                                            ArrayRef<Value>());
    rewriter.eraseOp(conditionTerminator);
    rewriter.inlineRegionBefore(conditionRegion, afterLoopBlock);

    // Emit body block
    Operation *bodyTerminator = bodyRegion.back().getTerminator();
    rewriter.eraseOp(bodyTerminator);
    rewriter.setInsertionPointToEnd(&bodyRegion.back());
    rewriter.create<mlir::cf::BranchOp>(loc, conditionBlock, std::nullopt);
    rewriter.inlineRegionBefore(bodyRegion, afterLoopBlock);

    // Emit branch to condition block
    rewriter.setInsertionPointToEnd(initialBlock);
    rewriter.create<mlir::cf::BranchOp>(loc, conditionBlock);

    rewriter.replaceOp(whileOp, afterLoopBlock->getArguments());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Func
//===----------------------------------------------------------------------===//

struct CallOpLowering : public ConversionPattern {

  CallOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::CallOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    auto callOp = dyn_cast<cherry::CallOp>(op);
    llvm::SmallVector<Type, 1> results;
    if (callOp.getResults().size() != 0)
      results.push_back(callOp.getResult(0).getType());

    rewriter.replaceOpWithNewOp<func::CallOp>(op, callOp.getCallee(), results,
                                              operands);
    return success();
  }
};

struct ReturnOpLowering : public ConversionPattern {
  ReturnOpLowering(MLIRContext *ctx)
      : ConversionPattern(cherry::ReturnOp::getOperationName(), 1, ctx) {}

  auto matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    if (operands.size() == 0) {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, std::nullopt);
    } else {
      Value operand = operands.front();
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, llvm::ArrayRef<Type>(),
                                                  operand);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertCherryToArithCfFunc
//===----------------------------------------------------------------------===//

struct ConvertCherryToArithCfFunc
    : public impl::ConvertCherryToArithCfFuncBase<ConvertCherryToArithCfFunc> {

  using impl::ConvertCherryToArithCfFuncBase<
      ConvertCherryToArithCfFunc>::ConvertCherryToArithCfFuncBase;

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                           func::FuncDialect, scf::SCFDialect>();
    target.addIllegalDialect<cherry::CherryDialect>();
    target.addLegalOp<cherry::CastOp>();
    target.addLegalOp<cherry::IfOp>();
    target.addLegalOp<cherry::PrintOp>();
    target.addLegalOp<cherry::YieldIfOp>();
    target.addLegalOp<cherry::StructInitOp>();
    target.addLegalOp<cherry::StructReadOp>();
    target.addLegalOp<cherry::StructWriteOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering, ArithmeticLogicOpLowering, WhileOpLowering,
                 CallOpLowering, ReturnOpLowering>(&getContext());

    auto f = getOperation();
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(f, target, patternSet))) {
      signalPassFailure();
    }
  }
};

} // end namespace
} // namespace mlir::cherry
