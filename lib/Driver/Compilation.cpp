//===--- Compilation.cpp - Compilation Task Data Structure ----------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Driver/Compilation.h"
#include "KaleidoscopeJIT.h"
#include "cherry/LLVMGen/LLVMGen.h"
#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "cherry/MLIRGen/MLIRGen.h"
#include "cherry/Parse/Lexer.h"
#include "cherry/Parse/Parser.h"
#include "cherry/Sema/Sema.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

using namespace cherry;

static auto makeContext() -> mlir::MLIRContext {
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  return mlir::MLIRContext(registry);
}

Compilation::Compilation() : _mlirContext{makeContext()} {}

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
  compilation->_mlirContext.getOrLoadDialect<mlir::arith::ArithDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::func::FuncDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::memref::MemRefDialect>();
  compilation->_mlirContext.getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir::registerBuiltinDialectTranslation(compilation->_mlirContext);
  mlir::registerLLVMDialectTranslation(compilation->_mlirContext);
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

auto Compilation::genMLIR(mlir::OwningOpRef<mlir::ModuleOp> &module,
                          Lowering lowering) -> CherryResult {
  std::unique_ptr<Module> moduleAST;
  if (parse(moduleAST))
    return failure();

  if (cherry::sema(_sourceManager, *moduleAST.get()) ||
      mlirGen(_sourceManager, _mlirContext, *moduleAST, module))
    return failure();

  mlir::PassManager pm(module.get()->getName());
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  if (_enableOpt)
    optPM.addPass(mlir::createCanonicalizerPass());

  if (lowering >= Lowering::SCF)
    optPM.addPass(mlir::cherry::createConvertCherryToSCF());

  if (lowering >= Lowering::ArithCfFunc)
    optPM.addPass(mlir::cherry::createConvertCherryToArithCfFunc());

  if (lowering >= Lowering::LLVM) {
    pm.addPass(mlir::cherry::createConvertCherryToLLVM());
    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
        mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }
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
    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (genMLIR(module, Lowering::LLVM))
      return failure();

    llvmModule = mlir::translateModuleToLLVMIR(*module, *this->_llvmContext);
    if (!llvmModule)
      return failure();

    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
      llvm::errs() << "Could not create JITTargetMachineBuilder\n";
      return failure();
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
      llvm::errs() << "Could not create TargetMachine\n";
      return failure();
    }
    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
        llvmModule.get(), tmOrError.get().get());

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
    uint64_t (*fp)() = symbol.getAddress().toPtr<uint64_t (*)()>();
    fp();
    return EXIT_SUCCESS;
  } else {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (genMLIR(module, Lowering::LLVM))
      return EXIT_FAILURE;

    auto optPipeline =
        mlir::makeOptimizingTransformer(_enableOpt ? 3 : 0,
                                        /*sizeLevel=*/0,
                                        /*targetMachine=*/nullptr);

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    int result;
    void *pResult = static_cast<void *>(&result);
    if (engine->invokePacked("main", {pResult})) {
      return EXIT_FAILURE;
    }
    return result;
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
  auto rm = std::optional<llvm::Reloc::Model>();
  auto targetMachine =
      target->createTargetMachine(targetTriple, cpu, features, opt, rm);

  llvmModule->setTargetTriple(targetTriple);
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  std::error_code ec;
  llvm::raw_fd_ostream dest(outputFileName, ec, llvm::sys::fs::OF_None);
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
  mlir::OwningOpRef<mlir::ModuleOp> module;
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