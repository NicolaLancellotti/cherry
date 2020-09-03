#include "cherry/Driver/Compilation.h"
#include "cherry/Parse/Lexer.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

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

  while (true) {
    auto token = lexer->lexToken();
    if (token.is(Token::eof)) {
      break;
    }
    auto [line, col] = _sourceManager.getLineAndColumn(token.getLoc());
    llvm::errs() << token.getTokenName() << " '" << token.getSpelling()
                 << "' Loc=<"<< line << ":" << col << ">\n";
  }
  return EXIT_SUCCESS;
}
