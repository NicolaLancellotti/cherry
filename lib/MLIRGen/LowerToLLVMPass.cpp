//===--- LowerToLLVMPass.cpp - Lowering to the llvm dialect ---------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/CherryOps.h"
#include "cherry/MLIRGen/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(cherry::PrintOp::getOperationName(), 1, context) {}

  auto matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const -> LogicalResult final {
    auto loc = op->getLoc();
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    // Get a symbol reference to the printf function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, parentModule, llvmDialect);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%d\n\0", 4), parentModule,
        llvmDialect);

    auto printOp = cast<cherry::PrintOp>(op);
    auto input = printOp.input();
    Value newInput = input.getType() == rewriter.getI64Type()
                               ? input : rewriter.create<LoadOp>(loc, input);

    rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                            ArrayRef<Value>({formatSpecifierCst, newInput}));
    rewriter.eraseOp(op);
    return success();
  }

private:

  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             LLVM::LLVMDialect *llvmDialect) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get("printf", context);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
    auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI32Ty, llvmI8PtrTy,
        /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get("printf", context);
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module,
                                       LLVM::LLVMDialect *llvmDialect) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(llvmDialect), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, LLVM::LLVMType::getInt64Ty(llvmDialect),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), globalPtr,
        ArrayRef<Value>({cst0, cst0}));
  }
};

struct CherryToLLVMLoweringPass
    : public PassWrapper<CherryToLLVMLoweringPass, OperationPass<ModuleOp>> {

  auto runOnOperation() -> void final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    LLVMTypeConverter typeConverter(&getContext());
    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    patterns.insert<PrintOpLowering>(&getContext());

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, patterns)))
      signalPassFailure();
  }
};

} // end namespace



namespace mlir {

auto mlir::cherry::createLowerToLLVMPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<CherryToLLVMLoweringPass>();
}
auto registerLowerToLLVMPass() -> void {
  PassRegistration<CherryToLLVMLoweringPass>("lower-cherry-std-to-llvm",
                                             "Lower Cherry and Standard operations into the LLVM dialect");
}

}
