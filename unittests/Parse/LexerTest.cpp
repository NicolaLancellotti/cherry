#include "cherry/Parse/Lexer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace cherry;

namespace {

TEST(LexerTest, firstTest) {
  auto input = R"(fn ; , { } ( ) & 01 a0 a0a)";
  auto inputBuffer = llvm::MemoryBuffer::getMemBuffer(input, "main.cherry");

  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(inputBuffer),
                                   /*IncludeLoc*/ llvm::SMLoc());

  auto lexer = std::make_unique<Lexer>(sourceManager);
  ASSERT_TRUE(lexer->lexToken().is(Token::kw_fn));
  ASSERT_TRUE(lexer->lexToken().is(Token::semi));
  ASSERT_TRUE(lexer->lexToken().is(Token::comma));
  ASSERT_TRUE(lexer->lexToken().is(Token::l_brace));
  ASSERT_TRUE(lexer->lexToken().is(Token::r_brace));
  ASSERT_TRUE(lexer->lexToken().is(Token::l_paren));
  ASSERT_TRUE(lexer->lexToken().is(Token::r_paren));
  ASSERT_TRUE(lexer->lexToken().is(Token::error));
  {
    auto token = lexer->lexToken();
    ASSERT_TRUE(token.is(Token::decimal));
    ASSERT_EQ(token.getSpelling(), "01");
    auto uint64 = token.getUInt64IntegerValue();
    ASSERT_TRUE(uint64.hasValue());
    ASSERT_EQ(uint64.getValue(), 1);
  }
  {
    auto token = lexer->lexToken();
    ASSERT_TRUE(token.is(Token::identifier));
    ASSERT_EQ(token.getSpelling(), "a0");
  }
  {
    auto token = lexer->lexToken();
    ASSERT_TRUE(token.is(Token::identifier));
    ASSERT_EQ(token.getSpelling(), "a0a");
  }
  ASSERT_TRUE(lexer->lexToken().is(Token::eof));
}

} // end namespace
