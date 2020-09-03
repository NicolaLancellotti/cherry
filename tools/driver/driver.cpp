#include "cherry/Driver/Compilation.h"
#include "llvm/Support/CommandLine.h"

using namespace cherry;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
namespace {
enum Action { None, DumpTokens };
}

static cl::opt<enum Action>
    emitAction("dump",
               cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpTokens,
                                     "tokens",
                                     "dump internal rep of tokens")));

auto main(int argc, const char **argv) -> int {
  cl::ParseCommandLineOptions(argc, argv, "Cherry compiler\n");

  switch (emitAction) {
  case Action::DumpTokens: {
    if (auto compilation = Compilation::make(inputFilename)) {
      return compilation->dumpTokens();
    }
    return EXIT_FAILURE;
  }
  default:
    return -1;
  }

}
