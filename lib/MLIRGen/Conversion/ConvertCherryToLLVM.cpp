//===--- ConvertCherryToLLVM.cpp ------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/CherryTypes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::cherry {

#define GEN_PASS_DEF_CONVERTCHERRYTOLLVM
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

class CastOpLowering : public ConversionPattern {
public:
  explicit CastOpLowering(MLIRContext *context)
      : ConversionPattern(cherry::CastOp::getOperationName(), 1, context) {}

  auto matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = op->getLoc();
    auto castOp = cast<cherry::CastOp>(op);
    auto operand = castOp.getInput();
    Value newOperand = llvm::isa<MemRefType>(operand.getType())
                           ? rewriter.create<memref::LoadOp>(loc, operand)
                           : operand;

    auto cast =
        rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI64Type(), newOperand);
    rewriter.replaceOp(op, cast.getRes());
    return mlir::success();
  }
};

class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(cherry::PrintOp::getOperationName(), 1, context) {}

  auto matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = op->getLoc();

    // Get a symbol reference to the printf function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%llu\n\0", 6), parentModule);

    auto printOp = cast<cherry::PrintOp>(op);
    auto operand = printOp.getInput();
    Value newOperand = llvm::isa<MemRefType>(operand.getType())
                           ? rewriter.create<memref::LoadOp>(loc, operand)
                           : operand;

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, printfRef, rewriter.getI64Type(),
        ArrayRef<Value>({formatSpecifierCst, newOperand}));
    return success();
  }

private:
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto i32Ty = IntegerType::get(context, 64); // TODO: cast
    auto i8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto fnType = LLVM::LLVMFunctionType::get(i32Ty, i8PtrTy,
                                              /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", fnType);
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

struct StructInitOpLowering : public ConvertOpToLLVMPattern<StructInitOp> {
  explicit StructInitOpLowering(LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  auto matchAndRewrite(StructInitOp op, StructInitOp::Adaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override final {
    auto loc = op->getLoc();

    SmallVector<Type> results;
    if (failed(typeConverter->convertTypes(op.getResult().getType(), results)))
      return failure();

    Value container = rewriter.create<mlir::LLVM::UndefOp>(loc, results);
    for (size_t i = 0; i < adaptor.getOperands().size(); ++i)
      container = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, container, adaptor.getOperands()[i], i);

    rewriter.replaceOp(op, container);
    return success();
  }
};

struct StructReadOpLowering : public ConvertOpToLLVMPattern<StructReadOp> {
  explicit StructReadOpLowering(LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  auto matchAndRewrite(StructReadOp op, StructReadOp::Adaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override final {
    SmallVector<int64_t> position = {static_cast<int64_t>(adaptor.getIndex())};
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, adaptor.getStructValue(), position);
    return success();
  }
};

struct StructWriteOpLowering : public ConvertOpToLLVMPattern<StructWriteOp> {
  explicit StructWriteOpLowering(LLVMTypeConverter &typeConverter,
                                 PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  auto matchAndRewrite(StructWriteOp op, StructWriteOp::Adaptor adaptor,
                       ConversionPatternRewriter &rewriter) const
      -> LogicalResult override final {
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, adaptor.getStructValue(), adaptor.getValueToStore(),
        adaptor.getIndexesAttr());
    return success();
  }
};

struct ConvertCherryToLLVM
    : public impl::ConvertCherryToLLVMBase<ConvertCherryToLLVM> {

  using impl::ConvertCherryToLLVMBase<
      ConvertCherryToLLVM>::ConvertCherryToLLVMBase;

  auto runOnOperation() -> void final {
    // Target
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    // Types conversions
    LLVMTypeConverter typeConverter(&getContext());

    typeConverter.addConversion([&](mlir::cherry::CherryStructType type) {
      SmallVector<Type, 2> types;
      for (auto t : type.getTypes()) {
        if (auto structType =
                llvm::dyn_cast<mlir::cherry::CherryStructType>(t)) {
          types.push_back(typeConverter.convertType(structType));
        } else {
          types.push_back(typeConverter.convertType(t));
        }
      }
      return LLVM::LLVMStructType::getLiteral(&getContext(), types);
    });

    // Patterns
    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<CastOpLowering, PrintOpLowering>(&getContext());
    patterns
        .add<StructInitOpLowering, StructReadOpLowering, StructWriteOpLowering>(
            typeConverter);

    // Conversion
    auto module = getOperation();
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyFullConversion(module, target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::cherry
