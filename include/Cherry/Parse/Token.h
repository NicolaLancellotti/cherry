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

  llvm::StringRef getSpelling() const { return spelling; }

  Kind getKind() const { return kind; }
  bool is(Kind K) const { return kind == K; }

  static llvm::Optional<uint64_t> getUInt64IntegerValue(llvm::StringRef spelling);
  llvm::Optional<uint64_t> getUInt64IntegerValue() const {
    return getUInt64IntegerValue(getSpelling());
  }

private:

  Kind kind;

  llvm::StringRef spelling;
};

}

#endif