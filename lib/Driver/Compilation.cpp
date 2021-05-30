//===--- Compilation.cpp - Compilation Task Data Structure ----------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Driver/Compilation.h"
#include "KaleidoscopeJIT.h"
#include "cherry/LLVMGen/LLVMGen.h"
#include "cherry/MLIRGen/CherryDialect.h"
#include "cherry/MLIRGen/MLIRGen.h"
#include "cherry/MLIRGen/Passes.h"
#include "cherry/Parse/Lexer.h"
#include "cherry/Parse/Parser.h"
#include "cherry/Sema/Sema.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace cherry;

auto Compilation::make(llvm::StringRef filename, bool enableOpt,
                       bool backendLLVM) -> std::unique_ptr<Compilation> {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);

  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "error: " << ec.message() << ": '" << filename << "'\n";
    return {};
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto compilation = std::make_unique<Compilation>();
  compilation->_mlirContext.getOrLoadDialect<mlir::cherry::CherryDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::StandardOpsDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::scf::SCFDialect>();
  compilation->_llvmContext = std::make_unique<llvm::LLVMContext>();
  compilation->sourceManager().AddNewSourceBuffer(std::move(fileOrErr.get()),
                                                  llvm::SMLoc());
  compilation->_enableOpt = enableOpt;
  compilation->_backendLLVM = backendLLVM;
  return compilation;
}

auto Compilation::parse(std::unique_ptr<Module> &module) -> CherryResult {
  auto lexer = std::make_unique<Lexer>(_sourceManager);
  auto parser = Parser{std::move(lexer), _sourceManager};
  return parser.parseModule(module);
}

auto Compilation::genMLIR(mlir::OwningModuleRef &module, Lowering lowering)
    -> CherryResult {
  std::unique_ptr<Module> moduleAST;
  if (parse(moduleAST))
    return failure();

  if (cherry::sema(_sourceManager, *moduleAST.get()) ||
      mlirGen(_sourceManager, _mlirContext, *moduleAST, module))
    return failure();

  mlir::PassManager pm(&_mlirContext);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  if (_enableOpt)
    optPM.addPass(mlir::createCanonicalizerPass());

  if (lowering >= Lowering::SCF)
    pm.addPass(mlir::cherry::createLowerToSCFPass());

  if (lowering >= Lowering::SCF_Standard)
    optPM.addPass(mlir::cherry::createLowerToSCFAndStandardPass());

  if (lowering >= Lowering::Standard)
    pm.addPass(mlir::createLowerToCFGPass());

  if (lowering >= Lowering::LLVM)
    pm.addPass(mlir::cherry::createLowerToLLVMPass());

  return pm.run(*module);
}

auto Compilation::genLLVM(std::unique_ptr<llvm::Module> &llvmModule)
    -> CherryResult {
  if (_backendLLVM) {
    std::unique_ptr<Module> module;
    if (parse(module) || cherry::sema(_sourceManager, *module.get()) ||
        llvmGen(_sourceManager, *_llvmContext, *module, llvmModule, _enableOpt))
      return failure();
  } else {
    mlir::OwningModuleRef module;
    if (genMLIR(module, Lowering::LLVM))
      return failure();

    llvmModule = mlir::translateModuleToLLVMIR(*module, *this->_llvmContext);
    if (!llvmModule)
      return failure();

    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    auto optPipeline =
        mlir::makeOptimizingTransformer(_enableOpt ? 3 : 0,
                                        /*sizeLevel=*/0,
                                        /*targetMachine=*/nullptr);

    if (auto err = optPipeline(llvmModule.get())) {
      llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
      return failure();
    }
  }

  return success();
}

auto Compilation::typecheck() -> int {
  std::unique_ptr<Module> module;
  if (parse(module) || cherry::sema(_sourceManager, *module.get()))
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

auto Compilation::jit() -> int {
  llvm::ExitOnError exitOnErr;

  if (_backendLLVM) {
    auto jit = exitOnErr(llvm::orc::KaleidoscopeJIT::Create());
    std::unique_ptr<llvm::Module> llvmModule;
    if (genLLVM(llvmModule))
      return EXIT_FAILURE;

    llvmModule->setDataLayout(jit->getDataLayout());

    exitOnErr(jit->addModule(llvm::orc::ThreadSafeModule(
        std::move(llvmModule), std::move(_llvmContext))));

    auto symbol = exitOnErr(jit->lookup("main"));
    uint64_t (*fp)() = (uint64_t(*)())(intptr_t)symbol.getAddress();
    fp();
    return EXIT_SUCCESS;
  } else {
    mlir::OwningModuleRef module;
    if (genMLIR(module, Lowering::LLVM))
      return EXIT_FAILURE;

    auto optPipeline =
        mlir::makeOptimizingTransformer(_enableOpt ? 3 : 0,
                                        /*sizeLevel=*/0,
                                        /*targetMachine=*/nullptr);

    auto maybeEngine = mlir::ExecutionEngine::create(
        *module, /*llvmModuleBuilder=*/nullptr, optPipeline);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    if (auto fun = engine->lookup("main")) {
      int result;
      void *pResult = (void *)&result;
      fun.get()(&pResult);
      return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
  }
}

auto Compilation::genObjectFile(const char *outputFileName) -> int {
  std::unique_ptr<llvm::Module> llvmModule;
  if (genLLVM(llvmModule))
    return EXIT_FAILURE;

  auto targetTriple = llvm::sys::getDefaultTargetTriple();

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << error;
    return EXIT_FAILURE;
  }

  auto cpu = "generic";
  auto features = "";
  llvm::TargetOptions opt;
  auto rm = llvm::Optional<llvm::Reloc::Model>();
  auto targetMachine =
      target->createTargetMachine(targetTriple, cpu, features, opt, rm);

  llvmModule->setTargetTriple(targetTriple);
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  std::error_code ec;
  llvm::raw_fd_ostream dest(outputFileName, ec, llvm::sys::fs::F_None);
  if (ec) {
    llvm::errs() << "Could not open file: " << ec.message();
    return EXIT_FAILURE;
  }

  llvm::legacy::PassManager pass;
  auto fileType = llvm::CGFT_ObjectFile;
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType))
    return EXIT_FAILURE;

  try {
    pass.run(*llvmModule);
  } catch (std::exception e) {
    llvm::errs() << e.what();
    return EXIT_FAILURE;
  }

  dest.flush();
  return EXIT_SUCCESS;
}

auto Compilation::dumpTokens() -> int {
  auto lexer = std::make_unique<Lexer>(_sourceManager);
  Lexer::tokenize(_sourceManager, *lexer);
  return EXIT_SUCCESS;
}

auto Compilation::dumpParse() -> int {
  std::unique_ptr<Module> module;
  if (parse(module))
    return EXIT_FAILURE;

  cherry::dumpAST(_sourceManager, *module);
  return EXIT_SUCCESS;
}

auto Compilation::dumpAST() -> int {
  std::unique_ptr<Module> module;
  if (parse(module) || cherry::sema(_sourceManager, *module.get()))
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