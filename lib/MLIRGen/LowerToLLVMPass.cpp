//===--- LowerToLLVMPass.cpp - Lowering to the llvm dialect ---------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "StructType.h"
#include "cherry/MLIRGen/CherryOps.h"
#include "cherry/MLIRGen/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

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
    auto operand = castOp.input();
    Value newOperand = operand.getType().isa<MemRefType>()
                           ? rewriter.create<LoadOp>(loc, operand)
                           : operand;

    auto cast =
        rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI64Type(), newOperand);
    rewriter.replaceOp(op, cast.res());
    return success();
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
    auto operand = printOp.input();
    Value newOperand = operand.getType().isa<MemRefType>()
                           ? rewriter.create<LoadOp>(loc, operand)
                           : operand;

    rewriter.replaceOpWithNewOp<CallOp>(
        op, printfRef, rewriter.getI64Type(),
        ArrayRef<Value>({formatSpecifierCst, newOperand}));
    return success();
  }

private:
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get("printf", context);

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
    return SymbolRefAttr::get("printf", context);
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

struct CherryToLLVMLoweringPass
    : public PassWrapper<CherryToLLVMLoweringPass, OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  auto runOnOperation() -> void final {
    // Target
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    // Patterns
    OwningRewritePatternList patterns;
    patterns.insert<PrintOpLowering>(&getContext());
    patterns.insert<CastOpLowering>(&getContext());

    // Types conversions
    LLVMTypeConverter typeConverter(&getContext());
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    typeConverter.addConversion([&](mlir::cherry::StructType type) {
      SmallVector<Type, 2> types;
      for (auto t : type.getElementTypes()) {
        if (t.isa<mlir::NoneType>()) {
          types.push_back(IntegerType::get(&getContext(), 1));
        } else if (auto structType = t.dyn_cast<mlir::cherry::StructType>()) {
          types.push_back(typeConverter.convertType(structType));
        } else {
          types.push_back(typeConverter.convertType(t));
        }
      }
      return LLVM::LLVMStructType::getLiteral(&getContext(), types);
    });

    // Conversion
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // end namespace

namespace mlir {

auto mlir::cherry::createLowerToLLVMPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<CherryToLLVMLoweringPass>();
}
auto registerLowerToLLVMPass() -> void {
  PassRegistration<CherryToLLVMLoweringPass>("convert-cherry-to-llvm",
                                             "Convert Cherry ops to llvm ops");
}

} // namespace mlir
