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

auto Parser::parseModule(unique_ptr<Module> &module) -> CherryResult {
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

auto Parser::parseDeclaration(unique_ptr<Decl> &decl) -> CherryResult {
  switch (tokenKind()) {
  case Token::kw_fn:
    return parseFunctionDecl_c(decl);
  case Token::kw_struct:
    return parseStructDecl_c(decl);
  default:
    return emitError(diag::expected_fun_struct);
  }
}

auto Parser::parseFunctionDecl_c(unique_ptr<Decl> &decl) -> CherryResult {
  auto loc = tokenLoc();
  unique_ptr<Prototype> proto;
  unique_ptr<BlockExpr> body;
  if (parsePrototype_c(proto) ||
      parseToken(Token::l_brace, diag::expected_l_brace)  ||
      parseBlockExpr(body))
    return failure();

  decl = make_unique<FunctionDecl>(loc, move(proto), move(body));
  return success();
}

auto Parser::parsePrototype_c(unique_ptr<Prototype> &proto) -> CherryResult {
  auto location = tokenLoc();
  consume(Token::kw_fn);

  // Parse name
  unique_ptr<Identifier> name;
  if (parseIdentifier(name, diag::expected_id) ||
      parseToken(Token::l_paren, diag::expected_l_paren))
    return failure();

  // Parse Element
  PE<VariableDeclExpr> parseParam = [this] (unique_ptr<VariableDeclExpr> &elem) -> CherryResult {
    unique_ptr<VariableExpr> param;
    unique_ptr<Identifier> type;
    if (parseIdentifier(param, diag::expected_id) ||
        parseToken(Token::colon, diag::expected_colon) ||
        parseIdentifier(type, diag::expected_type))
      return failure();
    elem = make_unique<VariableDeclExpr>(param->location(), move(param),
                                         move(type), nullptr);
    return success();
  };

  // Parse List
  VectorUniquePtr<VariableDeclExpr> parameters;
  unique_ptr<Identifier> type;
  if (parseList(Token::comma, Token::r_paren,
                diag::expected_comma_or_r_paren,
                diag::expected_r_paren, parameters, parseParam) ||
      parseToken(Token::colon, diag::expected_colon) ||
      parseIdentifier(type, diag::expected_type))
    return failure();

  // Make Proto
  proto = make_unique<Prototype>(location, move(name), move(parameters), move(type));
  return success();
}

auto Parser::parseBlockExpr(unique_ptr<BlockExpr> &block) -> CherryResult {
  auto loc = tokenLoc();
  VectorUniquePtr<Expr> expressions;
  while (true) {
    unique_ptr<Expr> expr;
    if (parseStatementWithoutSemi(expr))
      return failure();
    auto isStatement = expr->isStatement();
    expressions.push_back(move(expr));

    if (isStatement) {
      if (parseToken(Token::semi, diag::expected_semi))
        return failure();
      continue;
    }

    if (consumeIf(Token::semi))
      continue;

    if (parseToken(Token::r_brace, diag::expected_r_brace))
      return failure();

    block = make_unique<BlockExpr>(loc, std::move(expressions));
    return success();
  }
}

auto Parser::parseStructDecl_c(unique_ptr<Decl> &decl) -> CherryResult {
  auto loc = tokenLoc();
  consume(Token::kw_struct);

  // Parse Type
  unique_ptr<Identifier> type;
  if (parseIdentifier(type, diag::expected_id) ||
      parseToken(Token::l_brace, diag::expected_l_brace))
    return failure();

  // Parse Element
  PE<VariableDeclExpr> parseField = [this] (unique_ptr<VariableDeclExpr> &elem) -> CherryResult {
    unique_ptr<VariableExpr> var;
    unique_ptr<Identifier> type;
    if (parseIdentifier(var, diag::expected_id) ||
        parseToken(Token::colon, diag::expected_colon) ||
        parseIdentifier(type, diag::expected_type))
      return failure();
    elem = make_unique<VariableDeclExpr>(var->location(), move(var), move(type),
                                         nullptr);
    return success();
  };

  // Parse List
  VectorUniquePtr<VariableDeclExpr> fields;
  if (parseList(Token::comma, Token::r_brace,
                diag::expected_comma_or_r_brace,
                diag::expected_r_brace, fields, parseField))
    return failure();

  // Make StructDecl
  decl = make_unique<StructDecl>(loc, move(type), move(fields));
  return success();
}

template <typename T>
auto Parser::parseIdentifier(unique_ptr<T> &identifier,
                             const char * const message) -> CherryResult {
  auto location = tokenLoc();
  auto name = spelling();
  if (parseToken(Token::identifier, message))
    return failure();
  identifier = make_unique<T>(location, name);
  return success();
}

// _____________________________________________________________________________
// Parse Expressions

auto Parser::parseExpression(unique_ptr<Expr> &expr) -> CherryResult {
  if (parsePrimaryExpression(expr)) {
    return failure();
  } else {
    return parseBinaryExpRHS(0, expr);
  }
}

auto Parser::parseExpressions(VectorUniquePtr<Expr> &expressions,
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

auto Parser::parsePrimaryExpression(unique_ptr<Expr> &expr) -> CherryResult {
  switch (tokenKind()) {
  case Token::decimal:
    return parseDecimal_c(expr);
  case Token::identifier:
    return parseFuncStructVar_c(expr);
  case Token::kw_if:
    return parseIfExpr_c(expr);
  case Token::kw_true: {
    auto loc = tokenLoc();
    consume(Token::kw_true);
    expr = make_unique<BoolLiteralExpr>(loc, true);
    return success();
  }
  case Token::kw_false: {
    auto loc = tokenLoc();
    consume(Token::kw_false);
    expr = make_unique<BoolLiteralExpr>(loc, false);
    return success();
  }
  default:
    return emitError(diag::expected_expr);
  }
}

auto Parser::parseIfExpr_c(std::unique_ptr<Expr> &expr) -> CherryResult {
  auto loc = tokenLoc();
  consume(Token::kw_if);
  unique_ptr<Expr> condition;
  unique_ptr<BlockExpr> thenBlock;
  unique_ptr<BlockExpr> elseBlock;
  if (parseExpression(condition) ||
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(thenBlock) ||
      parseToken(Token::kw_else, diag::expected_else) ||
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(elseBlock))
    return failure();

  expr = make_unique<IfExpr>(loc, move(condition), move(thenBlock), std::move(elseBlock));
  return success();
}

auto Parser::parseDecimal_c(unique_ptr<Expr> &expr) -> CherryResult {
  auto loc = tokenLoc();
  if (auto value = token().getUInt64IntegerValue()) {
    consume(Token::decimal);
    expr = make_unique<DecimalLiteralExpr>(loc, move(*value));
    return success();
  }
  return emitError(diag::integer_literal_overflows);
}

auto Parser::parseFuncStructVar_c(unique_ptr<Expr> &expr) -> CherryResult {
  auto location = tokenLoc();
  auto name = spelling();
  consume(Token::identifier);
  switch (tokenKind()) {
  case Token::l_paren:
    return parseFunctionCall_c(location, name, expr);
  default:
    expr = make_unique<VariableExpr>(location, name);
    return success();
  }
}

auto Parser::parseFunctionCall_c(llvm::SMLoc location,
                                 llvm::StringRef name,
                                 unique_ptr<Expr> &expr) -> CherryResult {
  consume(Token::l_paren);
  auto expressions = VectorUniquePtr<Expr>();
  if (parseExpressions(expressions, Token::comma, Token::r_paren,
                       diag::expected_comma_or_r_paren,
                       diag::expected_r_paren))
    return failure();
  expr = make_unique<CallExpr>(location, name, move(expressions));
  return success();
}

auto Parser::parseBinaryExpRHS(int exprPrec, std::unique_ptr<Expr> &expr) -> CherryResult {
  while (true) {
    int tokPrec = getTokenPrecedence();
    if (tokPrec < exprPrec)
      return success();

    Token t = token();
    consume(t.getKind());
    auto location = tokenLoc();

    unique_ptr<Expr> rhs;
    if (parsePrimaryExpression(rhs))
      return emitError(diag::expected_expr);

    int nextPrec = getTokenPrecedence();
    bool rightAssociative = isTokenRightAssociative();
    if (tokPrec < nextPrec) {
      if (parseBinaryExpRHS(tokPrec + 1, rhs))
        return failure();
    } else if ((tokPrec == nextPrec) && rightAssociative) {
      if (parseBinaryExpRHS(tokPrec, rhs))
        return failure();
    }

    expr = std::make_unique<BinaryExpr>(location, t.getSpelling(),
                                        std::move(expr), std::move(rhs));
  }
}

auto Parser::getTokenPrecedence() -> int {
  switch (tokenKind()) {
  case Token::assign:
    return 2;
  case Token::dot:
    return 10;
  default:
    return -1;
  }
}

auto Parser::isTokenRightAssociative() -> bool {
  switch (tokenKind()) {
  case Token::assign:
    return true;
  case Token::dot:
    return false;
  default:
    return false;
  }
}

// _____________________________________________________________________________
// Parse Statements

auto Parser::parseStatementWithoutSemi(unique_ptr<Expr> &expr) -> CherryResult {
  switch (tokenKind()) {
  case Token::kw_var:
    return parseVarDecl_c(expr);
  default:
    return parseExpression(expr);
  }
}

auto Parser::parseVarDecl_c(unique_ptr<Expr> &expr) -> CherryResult {
  auto loc = tokenLoc();
  consume(Token::kw_var);
  unique_ptr<VariableExpr> var;
  unique_ptr<Identifier> type;
  unique_ptr<Expr> e;
  if (parseIdentifier(var, diag::expected_id) ||
      parseToken(Token::colon, diag::expected_colon) ||
      parseIdentifier(type, diag::expected_type) ||
      parseToken(Token::assign, diag::expected_assign) ||
      parseExpression(e))
    return failure();
  expr = make_unique<VariableDeclExpr>(loc, move(var), move(type), std::move(e));
  return success();
}