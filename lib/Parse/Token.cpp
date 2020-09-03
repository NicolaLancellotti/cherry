#include "cherry/Parse/Token.h"
#include "llvm/ADT/StringSwitch.h"

using namespace cherry;
using llvm::SMLoc;
using llvm::SMRange;

auto Token::getUInt64IntegerValue() const -> llvm::Optional<uint64_t> {
  uint64_t result = 0;
  if (spelling.getAsInteger(10, result))
    return llvm::None;
  return result;
}
SMLoc Token::getLoc() const { return SMLoc::getFromPointer(spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

auto Token::getTokenName() -> const char* {
  switch (kind) {
#define TOK_MARKER(NAME) case NAME:  return #NAME;
#define TOK_IDENTIFIER(NAME) case NAME:  return #NAME;
#define TOK_LITERAL(NAME) case NAME:  return #NAME;
#define TOK_PUNCTUATION(NAME, SPELLING) case NAME: return #NAME;
#define TOK_KEYWORD(SPELLING) case kw_##SPELLING:  return "kw_" #SPELLING;
#include "cherry/Parse/TokenKinds.def"
  default: return NULL;
  }
}