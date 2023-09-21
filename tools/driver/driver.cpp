//===--- driver.cpp - Cherry Compiler Driver ------------------------------===//

#include "cherry/Driver/Compilation.h"
#include "llvm/Support/CommandLine.h"

using namespace cherry;
namespace cl = llvm::cl;

namespace {
enum Action {
  None,
  Dump_Tokens,
  Dump_Parse,
  Dump_AST,
  Dump_MLIR,
  Dump_MLIR_SCF,
  Dump_MLIR_ARITH_CF_FUNC,
  Dump_MLIR_LLVM,
  Dump_LLVM,
};

enum Backend {
  MLIR,
  LLVM,
};
} // end namespace

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string>
    outputFilename("c", cl::desc("Generate a target \".o\" object file"),
                   cl::ValueOptional, cl::value_desc("filename"));

static cl::opt<bool> typecheck("typecheck",
                               cl::desc("Parse and type-check input file"));

static cl::opt<enum Action> dumpAction(
    "dump", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(Dump_Tokens, "tokens",
                          "dump internal rep of tokens")),
    cl::values(clEnumValN(Dump_Parse, "parse",
                          "parse and output the AST dump")),
    cl::values(clEnumValN(Dump_AST, "ast",
                          "parse, type-check and output the AST dump")),
    cl::values(clEnumValN(Dump_MLIR, "mlir", "output the MLIR dump (cherry)")),
    cl::values(clEnumValN(Dump_MLIR_SCF, "mlir1",
                          "output the MLIR dump (cherry + scf)")),
    cl::values(
        clEnumValN(Dump_MLIR_ARITH_CF_FUNC, "mlir2",
                   "output the MLIR dump (cherry + scf + arith + cf + func)")),
    cl::values(clEnumValN(Dump_MLIR_LLVM, "mlir-llvm",
                          "output the MLIR dump (llvm)")),
    cl::values(clEnumValN(Dump_LLVM, "llvm", "output the LLVM dump")));

static cl::opt<enum Backend>
    backend("b", cl::desc("Select the backend"), cl::init(MLIR),
            cl::values(clEnumValN(MLIR, "mlir", "select the MLIR backend")),
            cl::values(clEnumValN(LLVM, "llvm", "select the LLVM backend")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

auto main(int argc, const char **argv) -> int {
  cl::ParseCommandLineOptions(argc, argv, "Cherry compiler\n");

  std::unique_ptr<Compilation> compilation =
      Compilation::make(inputFilename, enableOpt, backend == Backend::LLVM);
  if (compilation == nullptr)
    return EXIT_FAILURE;

  if (outputFilename.getPosition())
    return compilation->genObjectFile(
        outputFilename != "" ? outputFilename.c_str() : "a.o");

  if (typecheck)
    return compilation->typecheck();

  switch (dumpAction) {
  case Action::Dump_Tokens:
    return compilation->dumpTokens();
  case Action::Dump_Parse:
    return compilation->dumpParse();
  case Action::Dump_AST:
    return compilation->dumpAST();
  case Action::Dump_MLIR:
    return compilation->dumpMLIR(Compilation::Lowering::None);
  case Action::Dump_MLIR_SCF:
    return compilation->dumpMLIR(Compilation::Lowering::SCF);
  case Action::Dump_MLIR_ARITH_CF_FUNC:
    return compilation->dumpMLIR(Compilation::Lowering::ArithCfFunc);
  case Action::Dump_MLIR_LLVM:
    return compilation->dumpMLIR(Compilation::Lowering::LLVM);
  case Action::Dump_LLVM:
    return compilation->dumpLLVM();
  default:
    return compilation->jit();
  }
}
