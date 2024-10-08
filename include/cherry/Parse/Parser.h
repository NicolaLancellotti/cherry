//===--- Parser.h - Cherry Language Parser ----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_PARSER_H
#define CHERRY_PARSER_H

#include "cherry/AST/AST.h"
#include "cherry/Basic/CherryResult.h"
#include "cherry/Parse/DiagnosticsParse.h"
#include "cherry/Parse/Lexer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace cherry {

using llvm::failure;
using llvm::success;

// Parse Element
template <typename T>
using PE = llvm::function_ref<CherryResult(std::unique_ptr<T> &)>;

class Parser {
public:
  Parser(std::unique_ptr<Lexer> lexer, llvm::SourceMgr &sourceManager)
      : _token{lexer->lexToken()}, _lexer{std::move(lexer)},
        _sourceManager{sourceManager} {}

  auto parseModule(std::unique_ptr<Module> &module) -> CherryResult;

private:
  Token _token;
  std::unique_ptr<Lexer> _lexer;
  llvm::SourceMgr &_sourceManager;

  // ___________________________________________________________________________
  // Lex

  auto token() -> Token & { return _token; }

  auto tokenIs(Token::Kind kind) -> bool { return token().is(kind); }

  auto tokenKind() -> Token::Kind { return token().getKind(); }

  auto tokenLoc() -> llvm::SMLoc { return token().getLoc(); }

  auto spelling() -> llvm::StringRef { return token().getSpelling(); }

  auto consume(Token::Kind kind) -> void {
    assert(token().is(kind) && "consumed an unexpected token");
    token() = _lexer->lexToken();
  }

  auto consumeIf(Token::Kind kind) -> bool {
    if (!token().is(kind))
      return false;
    consume(kind);
    return true;
  }

  // ___________________________________________________________________________
  // Error

  CherryResult emitError(const llvm::Twine &msg) {
    _sourceManager.PrintMessage(tokenLoc(), llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
  }

  // ___________________________________________________________________________

  auto parseToken(Token::Kind expected, const llvm::Twine &message)
      -> CherryResult {
    if (consumeIf(expected))
      return success();
    return emitError(message);
  }

  template <typename T>
  auto parseList(Token::Kind separator, Token::Kind end,
                 const char *const separator_error, const char *const end_error,
                 VectorUniquePtr<T> &elements, PE<T> parseElement)
      -> CherryResult;

  // _____________________________________________________________________________
  // Parse Identifiers

  auto parseType(std::unique_ptr<Type> &type) -> CherryResult;

  auto parseFunctionName(std::unique_ptr<FunctionName> &functionName,
                         const char *const message) -> CherryResult;

  auto parseUnitType(std::unique_ptr<Type> &unit) -> CherryResult;

  // ___________________________________________________________________________
  // Parse Declarations

  auto parseDeclaration(std::unique_ptr<Decl> &decl) -> CherryResult;

  auto parseFunctionDecl_c(std::unique_ptr<Decl> &decl) -> CherryResult;

  auto parsePrototype_c(std::unique_ptr<Prototype> &proto) -> CherryResult;

  auto parseBlockExpr(std::unique_ptr<BlockExpr> &block) -> CherryResult;

  auto parseStructDecl_c(std::unique_ptr<Decl> &elem) -> CherryResult;

  // ___________________________________________________________________________
  // Parse Expressions

  auto parseExpression(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseExpressions(VectorUniquePtr<Expr> &elem, Token::Kind separator,
                        Token::Kind end, const char *const separator_error,
                        const char *const end_error) -> CherryResult;

  auto parsePrimaryExpression(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseVariableExpr(std::unique_ptr<VariableExpr> &identifier)
      -> CherryResult;

  auto parseIfExpr_c(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseWhileExpr_c(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseDecimal_c(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseFuncStructVar_c(std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseFunctionCall_c(llvm::SMLoc location, llvm::StringRef name,
                           std::unique_ptr<Expr> &expr) -> CherryResult;

  auto parseBinaryExpRHS(int exprPrec, std::unique_ptr<Expr> &expr)
      -> CherryResult;
  auto getTokenPrecedence() -> int;
  auto isTokenRightAssociative() -> bool;
  auto tokenToOperator(Token token) -> BinaryExpr::Operator;

  // ___________________________________________________________________________
  // Parse Statements

  auto parseStatementWithoutSemi(std::unique_ptr<Stat> &stat) -> CherryResult;

  auto parseVarDecl_c(std::unique_ptr<Stat> &stat) -> CherryResult;
};

} // end namespace cherry

#endif // CHERRY_PARSER_H
