#ifndef CHERRY_TOKEN_H
#define CHERRY_TOKEN_H

#include "llvm/ADT/StringRef.h"

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

  Token(Kind kind, llvm::StringRef spelling) : kind(kind), spelling(spelling) {}

  auto getSpelling() const -> llvm::StringRef { return spelling; }

  auto getKind() const -> Kind { return kind; }
  auto is(Kind K) const -> bool { return kind == K; }

  auto getUInt64IntegerValue() const -> llvm::Optional<uint64_t>;

private:

  Kind kind;

  llvm::StringRef spelling;
};

}

#endif