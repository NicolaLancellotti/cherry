#include "cherry/Driver/Compilation.h"
#include "cherry/Parse/Lexer.h"
#include "cherry/Parse/Parser.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Dialect.h"
#include "cherry/IRGen/CherryDialect.h"
#include "cherry/IRGen/MLIRGen.h"
#include "mlir/IR/Module.h"

using namespace cherry;

auto Compilation::make(std::string filename) -> std::unique_ptr<Compilation> {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);

  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "error: " << ec.message() << ": '" << filename << "'\n";
    return {};
  }

  auto compilation = std::make_unique<Compilation>();
  compilation->sourceManager().AddNewSourceBuffer(std::move(fileOrErr.get()), llvm::SMLoc());
  return compilation;
}

auto Compilation::dumpTokens() -> int {
  auto lexer = std::make_unique<Lexer>(_sourceManager);
  Lexer::tokenize(_sourceManager, *lexer);
  return EXIT_SUCCESS;
}

auto Compilation::parse(std::unique_ptr<Module>& module) -> CherryResult {
  auto lexer = std::make_unique<Lexer>(_sourceManager);
  auto parser = Parser{std::move(lexer), _sourceManager};
  return parser.parseModule(module);
}

auto Compilation::dumpAST() -> int {
  std::unique_ptr<Module> module;
  if (parse(module))
    return EXIT_FAILURE;

  cherry::dumpAST(_sourceManager, *module);
  return EXIT_SUCCESS;
}

auto Compilation::dumpMLIR() -> int {
  mlir::registerDialect<mlir::cherry::CherryDialect>();
  mlir::MLIRContext context;

  std::unique_ptr<Module> moduleAST;
  if (parse(moduleAST))
    return EXIT_FAILURE;

  mlir::OwningModuleRef module = mlirGen(_sourceManager, context, *moduleAST);
  if (!module)
    return EXIT_FAILURE;

  module->dump();
  return EXIT_SUCCESS;
}