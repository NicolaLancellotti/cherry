//===--- Parser.cpp - Cherry Language Parser ---------------------------------//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "Parser.h"

using namespace cherry;
using std::make_pair;
using std::make_unique;
using std::move;
using std::unique_ptr;

auto Parser::parseModule(unique_ptr<Module>& module) -> CherryResult {
  auto loc = tokenLoc();
  VectorUniquePtr<Decl> declarations;
  do {
    unique_ptr<Decl> decl;
    if (parseDeclaration(decl))
      return failure();
    declarations.push_back(move(decl));
  } while (!tokenIs(Token::eof));

  module = make_unique<Module>(loc, move(declarations));
  return success();
}

template <typename T>
auto Parser::parseList(Token::Kind separator,
                       Token::Kind end,
                       const char * const separator_error,
                       const char * const end_error,
                       VectorUniquePtr<T> &elements,
                       PE<T> parseElement) -> CherryResult {
  while (!tokenIs(end) && !tokenIs(Token::eof)) {
    unique_ptr<T> exp;
    if (parseElement(exp))
      return failure();
    elements.push_back(move(exp));

    if (tokenIs(end))
      break;

    if (parseToken(separator, separator_error))
      return failure();
  }
  return parseToken(end, end_error);
}

// _____________________________________________________________________________
// Parse Declarations

auto Parser::parseDeclaration(unique_ptr<Decl>& decl) -> CherryResult {
  switch (tokenKind()) {
  case Token::kw_fun:
    return parseFunctionDecl_c(decl);
  case Token::kw_struct:
    return parseStructDecl_c(decl);
  default:
    return emitError(diag::expected_fun_struct);
  }
}

auto Parser::parseFunctionDecl_c(unique_ptr<Decl>& decl) -> CherryResult {
  auto loc = tokenLoc();
  unique_ptr<Prototype> proto;
  VectorUniquePtr<Expr> body;
  if (parsePrototype_c(proto),
      parseToken(Token::l_brace, diag::expected_l_brace),
      parseStatements(body, Token::semi, Token::r_brace,
                      diag::expected_semi, diag::expected_r_brace))
    return failure();

  decl = make_unique<FunctionDecl>(loc, move(proto), move(body));
  return success();
}

auto Parser::parsePrototype_c(unique_ptr<Prototype>& proto) -> CherryResult {
  auto location = tokenLoc();
  consume(Token::kw_fun);

  // Parse name
  unique_ptr<Identifier> name;
  if (parseIdentifier(name, diag::expected_id) ||
      parseToken(Token::l_paren, diag::expected_l_paren))
    return failure();

  // Parse Element
  PE<VariableDecl> parseParam = [this] (unique_ptr<VariableDecl> &elem) -> CherryResult {
    unique_ptr<VariableExpr> param;
    unique_ptr<Identifier> type;
    if (parseIdentifier(param, diag::expected_id) ||
        parseToken(Token::colon, diag::expected_colon) ||
        parseIdentifier(type, diag::expected_type))
      return failure();
    elem = make_unique<VariableDecl>(param->location(), move(param), move(type));
    return success();
  };

  // Parse List
  VectorUniquePtr<VariableDecl> parameters;
  if (parseList(Token::comma, Token::r_paren,
                diag::expected_comma_or_r_paren,
                diag::expected_r_paren, parameters, parseParam))
    return failure();

  // Make Proto
  proto = make_unique<Prototype>(location, move(name), move(parameters));
  return success();
}

auto Parser::parseStatements(VectorUniquePtr<Expr>& expressions,
                             Token::Kind separator,
                             Token::Kind end,
                             const char * const separator_error,
                             const char * const end_error) -> CherryResult {
  while (!tokenIs(end) && !tokenIs(Token::eof)) {
    unique_ptr<Expr> expr;
    if (parseExpression(expr) || parseToken(separator, separator_error))
      return failure();
    expressions.push_back(move(expr));
  }
  return parseToken(end, end_error);
}

auto Parser::parseStructDecl_c(unique_ptr<Decl>& decl) -> CherryResult {
  auto loc = tokenLoc();
  consume(Token::kw_struct);

  // Parse Type
  unique_ptr<Identifier> type;
  if (parseIdentifier(type, diag::expected_id) ||
      parseToken(Token::l_brace, diag::expected_l_brace))
    return failure();

  // Parse Element
  PE<VariableDecl> parseField = [this] (unique_ptr<VariableDecl> &elem) -> CherryResult {
    unique_ptr<VariableExpr> var;
    unique_ptr<Identifier> type;
    if (parseIdentifier(var, diag::expected_id) ||
        parseToken(Token::colon, diag::expected_colon) ||
        parseIdentifier(type, diag::expected_type))
      return failure();
    elem = make_unique<VariableDecl>(var->location(), move(var), move(type));
    return success();
  };

  // Parse List
  VectorUniquePtr<VariableDecl> fields;
  if (parseList(Token::comma, Token::r_brace,
                diag::expected_comma_or_r_brace,
                diag::expected_r_brace, fields, parseField))
    return failure();

  // Make StructDecl
  decl = make_unique<StructDecl>(loc, move(type), move(fields));
  return success();
}

template <typename T>
auto Parser::parseIdentifier(unique_ptr<T>& identifier,
                             const char * const message) -> CherryResult {
  auto location = tokenLoc();
  auto name = spelling();
  if (parseToken(Token::identifier, message))
    return failure();
  identifier = make_unique<T>(location, std::string(name));
  return success();
}

// _____________________________________________________________________________
// Parse Expressions

auto Parser::parseExpression(unique_ptr<Expr>& expr) -> CherryResult {
  switch (tokenKind()) {
  case Token::decimal:
    return parseDecimal_c(expr);
  case Token::identifier:
    return parseIdentifier_c(expr);
  default:
    return emitError(diag::expected_expr);
  }
}

auto Parser::parseExpressions(VectorUniquePtr<Expr>& expressions,
                              Token::Kind separator,
                              Token::Kind end,
                              const char * const separator_error,
                              const char * const end_error) -> CherryResult {
  PE<Expr> parseElement = [this] (unique_ptr<Expr> &elem) -> CherryResult {
    return parseExpression(elem);
  };
  return parseList(separator, end, separator_error, end_error,
                   expressions, parseElement);
}

auto Parser::parseDecimal_c(unique_ptr<Expr>& expr) -> CherryResult {
  auto loc = tokenLoc();
  if (auto value = token().getUInt64IntegerValue()) {
    consume(Token::decimal);
    expr = make_unique<DecimalExpr>(loc, move(*value));
    return success();
  }
  return emitError(diag::integer_literal_overflows);
}

auto Parser::parseIdentifier_c(unique_ptr<Expr>& expr) -> CherryResult {
  auto location = tokenLoc();
  std::string name{spelling()};
  consume(Token::identifier);
  switch (tokenKind()) {
  case Token::l_paren:
    return parseFunctionCall_c(location, name, expr);
  case Token::l_brace:
    return parseStructExpr_c(location, name, expr);
  default:
    expr = make_unique<VariableExpr>(location, move(name));
    return success();
  }
}

auto Parser::parseFunctionCall_c(llvm::SMLoc location,
                                 std::string name,
                                 unique_ptr<Expr>& expr) -> CherryResult {
  consume(Token::l_paren);
  auto expressions = VectorUniquePtr<Expr>();
  if (parseExpressions(expressions, Token::comma, Token::r_paren,
                       diag::expected_comma_or_r_paren,
                       diag::expected_r_paren))
    return failure();
  expr = make_unique<CallExpr>(location, move(name), move(expressions));
  return success();
}

auto Parser::parseStructExpr_c(llvm::SMLoc location,
                               std::string name,
                               unique_ptr<Expr>& expr) -> CherryResult {
  consume(Token::l_brace);
  auto expressions = VectorUniquePtr<Expr>();
  if (parseExpressions(expressions, Token::comma, Token::r_brace,
                       diag::expected_comma_or_r_brace,
                       diag::expected_r_brace))
    return failure();
  expr = make_unique<StructExpr>(location, move(name), move(expressions));
  return success();
}
