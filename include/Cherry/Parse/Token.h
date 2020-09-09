//===--- Token.h - Cherry Language Token ------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_TOKEN_H
#define CHERRY_TOKEN_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace cherry {

class Token {
public:
  enum Kind : int {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "TokenKinds.def"
  };

  Token(Kind kind, llvm::StringRef spelling) : _kind(kind), _spelling(spelling) {}

  auto getSpelling() const -> llvm::StringRef { return _spelling; }

  auto getKind() const -> Kind { return _kind; }
  auto is(Kind K) const -> bool { return _kind == K; }

  auto getUInt64IntegerValue() const -> llvm::Optional<uint64_t>;

  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

  auto getTokenName() -> const char*;

private:
  Kind _kind;
  llvm::StringRef _spelling;
};

} // end namespace cherry

#endif // CHERRY_TOKEN_H
