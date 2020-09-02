#ifndef CHERRY_LEXER_H
#define CHERRY_LEXER_H

#include "cherry/Parse/Token.h"
#include "llvm/Support/SourceMgr.h"

namespace cherry {

class Lexer {
public:

  explicit Lexer(const llvm::SourceMgr &sourceMgr);

  Token lexToken();

private:
  Token formToken(Token::Kind kind, const char *tokStart) {
    return Token(kind, llvm::StringRef(tokStart, curPtr - tokStart));
  }

  Token emitError(const char *loc, const llvm::Twine &message);

  Token lexKeyword(const char *tokStart);
  Token lexDecimal(const char *tokStart);

  const llvm::SourceMgr &sourceMgr;
  llvm::StringRef curBuffer;
  const char *curPtr;

  Lexer(const Lexer &) = delete;
  void operator=(const Lexer &) = delete;
};

}

#endif