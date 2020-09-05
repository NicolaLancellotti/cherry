#include "cherry/Parse/Lexer.h"
#include "mlir/IR/Diagnostics.h"

using namespace cherry;

Lexer::Lexer(const llvm::SourceMgr &sourceMgr): sourceMgr(sourceMgr) {
  auto bufferID = sourceMgr.getMainFileID();
  curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  curPtr = curBuffer.begin();
}

auto Lexer::lexToken() -> Token {
  while (true) {
    Restart:
    const char *tokStart = curPtr;
    switch (*curPtr++) {
    default:
      if (isalpha(curPtr[-1]))
        return lexIdentifierOrKeyword(tokStart);

      return formToken(Token::error, tokStart);
    case ' ':
    case '\t':
    case '\n':
    case '\r':
      // Handle whitespace.
      continue;
    case 0:
      if (curPtr - 1 == curBuffer.end())
        return formToken(Token::eof, tokStart);
      continue;
    case ';':
      return formToken(Token::semi, tokStart);
    case ',':
      return formToken(Token::comma, tokStart);
    case '(':
      return formToken(Token::l_paren, tokStart);
    case ')':
      return formToken(Token::r_paren, tokStart);
    case '{':
      return formToken(Token::l_brace, tokStart);
    case '}':
      return formToken(Token::r_brace, tokStart);
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexDecimal(tokStart);
    case '#': {
      while (true) {
        if (*curPtr == '\n' || *curPtr == 0) {
          goto Restart;
        }
        curPtr++;
      }
    }
    }
  }
}

auto Lexer::lexIdentifierOrKeyword(const char *tokStart) -> Token {
  // Match [0-9a-zA-Z]*
  while (isalpha(*curPtr) || isdigit(*curPtr))
    ++curPtr;

  llvm::StringRef spelling(tokStart, curPtr - tokStart);

  Token::Kind kind = llvm::StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, Token::kw_##SPELLING)
#include "TokenKinds.def"
      .Default(Token::identifier);

  return Token(kind, spelling);
}

auto Lexer::lexDecimal(const char *tokStart) -> Token {
  while (isdigit(*curPtr))
    ++curPtr;

  return formToken(Token::decimal, tokStart);
}