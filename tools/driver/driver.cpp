//===--- driver.cpp - Cherry Compiler Driver ------------------------------===//

#include "cherry/Driver/Compilation.h"
#include "llvm/Support/CommandLine.h"

using namespace cherry;
namespace cl = llvm::cl;

namespace {
enum Action {
  None,
  DumpTokens,
  DumpParse,
  DumpAST,
  DumpMLIR,
  DumpMLIRStandard,
  DumpMLIRLLVM,
  DumpLLVM,
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

static cl::opt<std::string> outputFilename("c",
                                           cl::desc("Generate a target \".o\" object file"),
                                           cl::ValueOptional,
                                           cl::value_desc("filename"));

static cl::opt<bool> typecheck("typecheck",
                               cl::desc("Parse and type-check input file"));

static cl::opt<enum Action>
    dumpAction("dump",
               cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpTokens,
                                     "tokens",
                                     "dump internal rep of tokens")),
               cl::values(clEnumValN(DumpParse,
                                     "parse",
                                     "parse and output the AST dump")),
               cl::values(clEnumValN(DumpAST,
                                     "ast",
                                     "parse, type-check and output the AST dump")),
               cl::values(clEnumValN(DumpMLIR,
                                     "mlir",
                                     "output the MLIR dump")),
               cl::values(clEnumValN(DumpMLIRStandard,
                                     "mlir-std",
                                     "output the MLIR dump after std lowering")),
               cl::values(clEnumValN(DumpMLIRLLVM,
                                     "mlir-llvm",
                                     "output the MLIR dump after std and llvm lowering")),
               cl::values(clEnumValN(DumpLLVM,
                                     "llvm",
                                     "output the LLVM dump")));

static cl::opt<enum Backend>
    backend("b",
            cl::desc("Select the backend"),
            cl::init(MLIR),
            cl::values(clEnumValN(MLIR,
                                  "mlir",
                                  "select the MLIR backend")),
            cl::values(clEnumValN(LLVM,
                                  "llvm",
                                  "select the LLVM backend")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

auto main(int argc, const char **argv) -> int {
  cl::ParseCommandLineOptions(argc, argv, "Cherry compiler\n");

  std::unique_ptr<Compilation> compilation = Compilation::make(inputFilename,
                                                               enableOpt,
                                                               backend == Backend::LLVM);
  if (compilation == nullptr)
    return EXIT_FAILURE;

  if (outputFilename.getPosition())
    return compilation->genObjectFile(outputFilename != "" ? outputFilename.c_str() : "a.o");

  if (typecheck)
    return compilation->typecheck();

  switch (dumpAction) {
  case Action::DumpTokens:
    return compilation->dumpTokens();
  case Action::DumpParse:
    return compilation->dumpParse();
  case Action::DumpAST:
    return compilation->dumpAST();
  case Action::DumpMLIR:
    return compilation->dumpMLIR(Compilation::Lowering::None);
  case Action::DumpMLIRStandard:
    return compilation->dumpMLIR(Compilation::Lowering::Standard);
  case Action::DumpMLIRLLVM:
    return compilation->dumpMLIR(Compilation::Lowering::LLVM);
  case Action::DumpLLVM:
    return compilation->dumpLLVM();
  default:
    return compilation->jit();
  }

}
