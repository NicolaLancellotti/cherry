#include "cherry/Driver/Compilation.h"
#include "llvm/Support/CommandLine.h"

using namespace cherry;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
namespace {
enum Action { None, DumpTokens, DumpAST };
}

static cl::opt<enum Action>
    emitAction("dump",
               cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpTokens,
                                     "tokens",
                                     "dump internal rep of tokens")),
               cl::values(clEnumValN(DumpAST,
                                     "ast",
                                     "dump the AST")));

auto main(int argc, const char **argv) -> int {
  cl::ParseCommandLineOptions(argc, argv, "Cherry compiler\n");

  std::unique_ptr<Compilation> compilation = Compilation::make(inputFilename);
  if (compilation == nullptr ) {
    return EXIT_FAILURE;
  }

  switch (emitAction) {
  case Action::DumpTokens:
      return compilation->dumpTokens();
  case Action::DumpAST:
    return compilation->dumpAST();
  default:
    return -1;
  }

}
