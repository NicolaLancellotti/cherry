//===--- Token.cpp - Cherry Language Token --------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "Token.h"
#include "llvm/ADT/StringSwitch.h"

using namespace cherry;
using llvm::SMLoc;
using llvm::SMRange;

auto Token::getUInt64IntegerValue() const -> llvm::Optional<uint64_t> {
  uint64_t result = 0;
  if (_spelling.getAsInteger(10, result))
    return llvm::None;
  return result;
}
SMLoc Token::getLoc() const { return SMLoc::getFromPointer(_spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(_spelling.data() + _spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

auto Token::getTokenName() -> const char* {
  switch (_kind) {
#define TOK_MARKER(NAME) case NAME:  return #NAME;
#define TOK_IDENTIFIER(NAME) case NAME:  return #NAME;
#define TOK_LITERAL(NAME) case NAME:  return #NAME;
#define TOK_PUNCTUATION(NAME, SPELLING) case NAME: return #NAME;
#define TOK_KEYWORD(SPELLING) case kw_##SPELLING:  return "kw_" #SPELLING;
#include "cherry/Parse/TokenKinds.def"
  default: return NULL;
  }
}
