#ifndef CHERRY_LEXER_H
#define CHERRY_LEXER_H

#include "cherry/Parse/Token.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace cherry {

class Lexer {
public:

  explicit Lexer(const llvm::SourceMgr &sourceMgr);

  auto lexToken() -> Token;

  static auto tokenize(const llvm::SourceMgr &sourceManager, Lexer& lexer) -> void {
    while (true) {
      auto token = lexer.lexToken();
      if (token.is(Token::eof)) {
        break;
      }
      auto [line, col] = sourceManager.getLineAndColumn(token.getLoc());
      llvm::errs() << token.getTokenName() << " '" << token.getSpelling()
                   << "' loc="<< line << ":" << col << "\n";
    }
  }

private:
  auto formToken(Token::Kind kind, const char *tokStart) -> Token {
    return Token(kind, llvm::StringRef(tokStart, curPtr - tokStart));
  }

  auto lexIdentifierOrKeyword(const char *tokStart) -> Token;
  auto lexDecimal(const char *tokStart) -> Token;

  const llvm::SourceMgr &sourceMgr;
  llvm::StringRef curBuffer;
  const char *curPtr;

  Lexer(const Lexer &) = delete;
  auto operator=(const Lexer &) -> void = delete;
};

}

#endif