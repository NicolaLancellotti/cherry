//===--- Compilation.cpp - Compilation Task Data Structure ----------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Driver/Compilation.h"
#include "cherry/MLIRGen/CherryDialect.h"
#include "cherry/MLIRGen/MLIRGen.h"
#include "cherry/MLIRGen/Passes.h"
#include "cherry/Parse/Lexer.h"
#include "cherry/Parse/Parser.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"

using namespace cherry;

auto Compilation::make(std::string filename,
                       bool enableOpt) -> std::unique_ptr<Compilation> {
  mlir::registerDialect<mlir::cherry::CherryDialect>();
  mlir::registerDialect<mlir::StandardOpsDialect>();
  mlir::registerDialect<mlir::LLVM::LLVMDialect>();

  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);

  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "error: " << ec.message() << ": '" << filename << "'\n";
    return {};
  }

  auto compilation = std::make_unique<Compilation>();
  compilation->sourceManager().AddNewSourceBuffer(std::move(fileOrErr.get()), llvm::SMLoc());
  compilation->_enableOpt = enableOpt;
  return compilation;
}

auto Compilation::parse(std::unique_ptr<Module>& module) -> CherryResult {
  auto lexer = std::make_unique<Lexer>(_sourceManager);
  auto parser = Parser{std::move(lexer), _sourceManager};
  return parser.parseModule(module);
}

auto Compilation::genMLIR(mlir::OwningModuleRef& module,
                          Lowering lowering) -> CherryResult {
  std::unique_ptr<Module> moduleAST;
  if (parse(moduleAST))
    return failure();

  module = mlirGen(_sourceManager, _context, *moduleAST);
  if (!module)
    return failure();

  mlir::PassManager pm(&_context);
  if (_enableOpt) {
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
  }

  if (lowering >= Lowering::Standard)
    pm.addPass(mlir::cherry::createLowerToStandardPass());

  if (lowering >= Lowering::LLVM)
    pm.addPass(mlir::cherry::createLowerToLLVMPass());

  return pm.run(*module);
}

auto Compilation::genLLVM(std::unique_ptr<llvm::Module>& llvmModule) -> CherryResult {
  mlir::OwningModuleRef module;
  if (genMLIR(module, Lowering::LLVM))
    return failure();

  llvmModule = mlir::translateModuleToLLVMIR(*module);
  if (!llvmModule) {
    return failure();
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  auto optPipeline = mlir::makeOptimizingTransformer(
      _enableOpt ? 3 : 0,
      /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return failure();
  }

  return success();
}

auto Compilation:: jit() -> int {
  mlir::OwningModuleRef module;
  if (genMLIR(module, Lowering::LLVM))
    return EXIT_FAILURE;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto optPipeline = mlir::makeOptimizingTransformer(
      _enableOpt ? 3 : 0,
      /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  auto maybeEngine = mlir::ExecutionEngine::create(*module, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  if (auto fun = engine->lookup("main")) {
    int result;
    void *pResult = (void*)&result;
    fun.get()(&pResult);
    return EXIT_SUCCESS;
  }

  return EXIT_SUCCESS;
}

auto Compilation::dumpTokens() -> int {
  auto lexer = std::make_unique<Lexer>(_sourceManager);
  Lexer::tokenize(_sourceManager, *lexer);
  return EXIT_SUCCESS;
}

auto Compilation::dumpAST() -> int {
  std::unique_ptr<Module> module;
  if (parse(module))
    return EXIT_FAILURE;

  cherry::dumpAST(_sourceManager, *module);
  return EXIT_SUCCESS;
}

auto Compilation::dumpMLIR(Lowering lowering) -> int {
  mlir::OwningModuleRef module;
  if (genMLIR(module, lowering))
    return EXIT_FAILURE;

  module->dump();
  return EXIT_SUCCESS;
}

auto Compilation::dumpLLVM() -> int {
  std::unique_ptr<llvm::Module> llvmModule;
  if (genLLVM(llvmModule))
    return EXIT_FAILURE;

  llvm::errs() << *llvmModule << "\n";
  return EXIT_SUCCESS;
}
