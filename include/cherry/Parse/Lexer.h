//===--- Lexer.h - Cherry Language Lexer ------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_LEXER_H
#define CHERRY_LEXER_H

#include "Token.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace cherry {

class Lexer {
public:
  explicit Lexer(const llvm::SourceMgr &sourceMgr);

  auto lexToken() -> Token;

  static auto tokenize(const llvm::SourceMgr &sourceManager, Lexer &lexer)
      -> void {
    while (true) {
      auto token = lexer.lexToken();
      if (token.is(Token::eof)) {
        break;
      }
      auto lineCol = sourceManager.getLineAndColumn(token.getLoc());
      auto line = std::get<0>(lineCol);
      auto col = std::get<1>(lineCol);
      llvm::errs() << token.getTokenName() << " '" << token.getSpelling()
                   << "' loc=" << line << ":" << col << "\n";
    }
  }

private:
  auto formToken(Token::Kind kind, const char *tokStart) -> Token {
    return Token(kind, llvm::StringRef(tokStart, _curPtr - tokStart));
  }

  auto lexIdentifierOrKeyword(const char *tokStart) -> Token;
  auto lexDecimal(const char *tokStart) -> Token;

  llvm::StringRef _curBuffer;
  const char *_curPtr;

  Lexer(const Lexer &) = delete;
  auto operator=(const Lexer &) -> void = delete;
};

} // end namespace cherry

#endif // CHERRY_LEXER_H
