//===--- Parser.cpp - Cherry Language Parser --------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
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

using mlir::failure;
using mlir::success;
using std::make_pair;

class Parser {
public:
  Parser(std::unique_ptr<Lexer> lexer,
         llvm::SourceMgr& sourceManager)
      : _token{lexer->lexToken()},
        _lexer{std::move(lexer)},
        _sourceManager{sourceManager} {}

  auto parseModule(std::unique_ptr<Module>& module) -> CherryResult {
    auto loc = tokenLoc();
    VectorUniquePtr<Decl> declarations;
    do {
      std::unique_ptr<Decl> decl;
      if (parseDeclaration(decl))
        return failure();
      declarations.push_back(std::move(decl));
    } while (!tokenIs(Token::eof));

    module = std::make_unique<Module>(loc, std::move(declarations));
    return success();
  }

private:
  Token _token;
  std::unique_ptr<Lexer> _lexer;
  llvm::SourceMgr& _sourceManager;

  // ___________________________________________________________________________
  // Lex

  auto token() -> Token& { return _token; }
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
    _sourceManager.PrintMessage(tokenLoc(),
                                llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
  }

  // ___________________________________________________________________________
  // Parse Token

  auto parseToken(Token::Kind expected,
                  const llvm::Twine &message) -> CherryResult {
    if (consumeIf(expected))
      return success();
    return emitError(message);
  }

  // ___________________________________________________________________________
  // Parse Declarations

  auto parseDeclaration(std::unique_ptr<Decl>& decl) -> CherryResult {
    switch (tokenKind()) {
    case Token::kw_fun: {
      std::unique_ptr<Decl> func;
      if (parseFunctionDecl_c(func))
        return failure();
      decl = std::move(func);
      return success();
    }
    case Token::kw_struct: {
      std::unique_ptr<Decl> structType;
      if (parseStructDecl_c(structType))
        return failure();
      decl = std::move(structType);
      return success();
    }
    default:
      return emitError(diag::expected_fun_struct);
    }
  }

  template <typename T>
  auto parseIdentifier(std::unique_ptr<T>& identifier, const llvm::Twine &message) -> CherryResult {
    auto location = tokenLoc();
    auto name = spelling();
    if (parseToken(Token::identifier, message))
      return failure();

    identifier = std::make_unique<T>(location, std::string(name));
    return success();
  }

  // Parse Functions

  auto parseFunctionDecl_c(std::unique_ptr<Decl>& decl) -> CherryResult {
    auto loc = tokenLoc();
    std::unique_ptr<Prototype> proto;
    VectorUniquePtr<Expr> body;

    if (parsePrototype_c(proto) || parseFunctionBody(body))
      return failure();

    decl = std::make_unique<FunctionDecl>(loc, std::move(proto), std::move(body));

    return success();
  }

  auto parsePrototype_c(std::unique_ptr<Prototype>& proto) -> CherryResult {
    auto location = tokenLoc();
    consume(Token::kw_fun);

    std::unique_ptr<Identifier> name;
    if (parseIdentifier(name, diag::expected_id) ||
        parseToken(Token::l_paren, diag::expected_l_paren_in_arg_list))
      return failure();

    // Parse parameters
    VectorUniquePtr<VariableDecl> parameters;
    while (!tokenIs(Token::r_paren) && !tokenIs(Token::eof)) {
      std::unique_ptr<VariableExpr> param;
      std::unique_ptr<Identifier> type;
      if (parseIdentifier(param, diag::expected_id) ||
          parseToken(Token::colon, diag::expected_colon) ||
          parseIdentifier(type, diag::expected_type))
        return failure();
      parameters.push_back(std::make_unique<VariableDecl>(param->location(),
                                                          std::move(param),
                                                          std::move(type)));
      if (tokenIs(Token::r_paren))
        break;

      if (parseToken(Token::comma,
                     diag::expected_comma_or_r_paren_arg_list))
        return failure();
    }

    consume(Token::r_paren);
    proto = std::make_unique<Prototype>(location, std::move(name), std::move(parameters));
    return success();
  }

  // Parse Types

  auto parseStructDecl_c(std::unique_ptr<Decl>& decl) -> CherryResult {
    auto loc = tokenLoc();
    consume(Token::kw_struct);

    std::unique_ptr<Identifier> name;
    if (parseIdentifier(name, diag::expected_id) ||
        parseToken(Token::l_brace, diag::expected_l_brace))
      return failure();

    VectorUniquePtr<VariableDecl> variableAndTypes;
    while (!tokenIs(Token::r_brace) && !tokenIs(Token::eof)) {
      std::unique_ptr<VariableExpr> var;
      std::unique_ptr<Identifier> type;
      if (parseIdentifier(var, diag::expected_id) ||
          parseToken(Token::colon, diag::expected_colon) ||
          parseIdentifier(type, diag::expected_type))
        return failure();
      variableAndTypes.push_back(std::make_unique<VariableDecl>(var->location(),
                                                                std::move(var),
                                                                std::move(type)));
      if (tokenIs(Token::r_brace))
        break;

      if (parseToken(Token::comma,
                     diag::expected_comma_or_r_brace))
        return failure();
    }

    consume(Token::r_brace);
    decl = std::make_unique<StructDecl>(loc, std::move(name), std::move(variableAndTypes));

    return success();
  }

  // ___________________________________________________________________________
  // Parse Expressions

  auto parseFunctionBody(VectorUniquePtr<Expr>& expressions) -> CherryResult {
    if (parseToken(Token::l_brace, diag::expected_l_brace_func_body))
      return failure();

    while (!tokenIs(Token::r_brace) && !tokenIs(Token::eof)) {
      std::unique_ptr<Expr> expr;
      if (parseExpression(expr) ||
          parseToken(Token::semi, diag::expected_semi)) {
        return failure();
      }
      expressions.push_back(std::move(expr));
    }

    if (parseToken(Token::r_brace, diag::expected_r_brace_func_body))
      return failure();

    return success();
  }

  auto parseExpression(std::unique_ptr<Expr>& expr) -> CherryResult {
    switch (tokenKind()) {
    case Token::decimal: {
      std::unique_ptr<DecimalExpr> decimal;
      if (parseDecimal_c(decimal))
        return failure();
      expr = std::move(decimal);
      return success();
    }
    case Token::identifier: {
      std::unique_ptr<Expr> var_or_fun;
      if (parseVariableOrFunctionCall_c(var_or_fun))
        return failure();
      expr = std::move(var_or_fun);
      return success();
    }
    default:
      return emitError(diag::expected_expr);
    }
  }

  auto parseDecimal_c(std::unique_ptr<DecimalExpr>& expr) -> CherryResult {
    auto loc = tokenLoc();
    if (auto value = token().getUInt64IntegerValue()) {
      consume(Token::decimal);
      expr = std::make_unique<DecimalExpr>(loc, std::move(*value));
      return success();
    }
    return emitError(diag::integer_literal_overflows);
  }

  auto parseVariableOrFunctionCall_c(std::unique_ptr<Expr>& expr) -> CherryResult {
    auto location = tokenLoc();
    std::string name{spelling()};
    consume(Token::identifier);

    if (tokenIs(Token::l_paren)) {
      std::unique_ptr<CallExpr> callExp;
      if (parseFunctionCall_c(location, name, callExp))
        return failure();
      expr = std::move(callExp);
      return success();
    }

    expr = std::make_unique<VariableExpr>(location, std::move(name));
    return success();
  }

  auto parseFunctionCall_c(llvm::SMLoc location,
                           std::string name,
                           std::unique_ptr<CallExpr>& expr) -> CherryResult {
    consume(Token::l_paren);
    auto expressions = VectorUniquePtr<Expr>();
    while (!tokenIs(Token::r_paren)) {
      std::unique_ptr<Expr> exp;
      if (parseExpression(exp))
        return failure();
      expressions.push_back(std::move(exp));
      if (tokenIs(Token::r_paren))
        break;

      if (parseToken(Token::comma,
                     diag::expected_comma_or_r_paren_arg_list))
        return failure();
    }

    consume(Token::r_paren);
    expr = std::make_unique<CallExpr>(location, std::move(name),
                                      std::move(expressions));
    return success();
  }

};

} // end namespace cherry

#endif // CHERRY_PARSER_H
