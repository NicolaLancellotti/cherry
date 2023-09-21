//===--- Parser.cpp - Cherry Language Parser ---------------------------------//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Parse/Parser.h"

using namespace cherry;
using std::make_unique;
using std::unique_ptr;

auto Parser::parseModule(unique_ptr<Module> &module) -> CherryResult {
  auto loc = tokenLoc();
  VectorUniquePtr<Decl> declarations;
  do {
    unique_ptr<Decl> decl;
    if (parseDeclaration(decl))
      return failure();
    declarations.push_back(std::move(decl));
  } while (!tokenIs(Token::eof));

  module = make_unique<Module>(loc, std::move(declarations));
  return success();
}

template <typename T>
auto Parser::parseList(Token::Kind separator, Token::Kind end,
                       const char *const separator_error,
                       const char *const end_error,
                       VectorUniquePtr<T> &elements, PE<T> parseElement)
    -> CherryResult {
  while (!tokenIs(end) && !tokenIs(Token::eof)) {
    unique_ptr<T> exp;
    if (parseElement(exp))
      return failure();
    elements.push_back(std::move(exp));

    if (tokenIs(end))
      break;

    if (parseToken(separator, separator_error))
      return failure();
  }
  return parseToken(end, end_error);
}

// _____________________________________________________________________________
// Parse Identifiers

auto Parser::parseUnitType(unique_ptr<Type> &unit) -> CherryResult {
  auto location = tokenLoc();
  consume(Token::l_paren);
  if (parseToken(Token::r_paren, diag::expected_l_paren))
    return failure();
  unit = make_unique<Type>(location, "Unit");
  return success();
}

auto Parser::parseType(unique_ptr<Type> &type) -> CherryResult {
  if (tokenIs(Token::l_paren))
    return parseUnitType(type);

  auto location = tokenLoc();
  auto name = spelling();
  if (parseToken(Token::identifier, diag::expected_type))
    return failure();
  type = make_unique<Type>(location, name);
  return success();
}

auto Parser::parseFunctionName(unique_ptr<FunctionName> &functionName,
                               const char *const message) -> CherryResult {
  auto location = tokenLoc();
  auto name = spelling();
  if (parseToken(Token::identifier, message))
    return failure();
  functionName = make_unique<FunctionName>(location, name);
  return success();
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
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(body))
    return failure();

  decl = make_unique<FunctionDecl>(loc, std::move(proto), std::move(body));
  return success();
}

auto Parser::parsePrototype_c(unique_ptr<Prototype> &proto) -> CherryResult {
  auto location = tokenLoc();
  consume(Token::kw_fn);

  // Parse name
  unique_ptr<FunctionName> name;
  if (parseFunctionName(name, diag::expected_id) ||
      parseToken(Token::l_paren, diag::expected_l_paren))
    return failure();

  // Parse Element
  PE<VariableStat> parseParam =
      [this](unique_ptr<VariableStat> &elem) -> CherryResult {
    unique_ptr<VariableExpr> param;
    unique_ptr<Type> type;
    if (parseVariableExpr(param) ||
        parseToken(Token::colon, diag::expected_colon) || parseType(type))
      return failure();
    elem = make_unique<VariableStat>(param->location(), std::move(param),
                                     std::move(type), nullptr);
    return success();
  };

  // Parse List
  VectorUniquePtr<VariableStat> parameters;
  unique_ptr<Type> type;
  if (parseList(Token::comma, Token::r_paren, diag::expected_comma_or_r_paren,
                diag::expected_r_paren, parameters, parseParam) ||
      parseToken(Token::colon, diag::expected_colon) || parseType(type))
    return failure();

  // Make Proto
  proto = make_unique<Prototype>(location, std::move(name),
                                 std::move(parameters), std::move(type));
  return success();
}

auto Parser::parseBlockExpr(unique_ptr<BlockExpr> &block) -> CherryResult {
  auto loc = tokenLoc();
  VectorUniquePtr<Stat> statements;
  while (true) {
    unique_ptr<Stat> stat;
    if (parseStatementWithoutSemi(stat))
      return failure();

    auto isStatement = stat->getKind() != Stat::Stat_Expression;
    if (isStatement) {
      if (parseToken(Token::semi, diag::expected_semi))
        return failure();
      statements.push_back(std::move(stat));
      continue;
    }

    if (consumeIf(Token::semi)) {
      statements.push_back(std::move(stat));
      continue;
    }

    if (parseToken(Token::r_brace, diag::expected_r_brace))
      return failure();

    unique_ptr<ExprStat> exprStat(static_cast<ExprStat *>(stat.release()));
    block = make_unique<BlockExpr>(loc, std::move(statements),
                                   std::move(exprStat->expression()));
    return success();
  }
}

auto Parser::parseStructDecl_c(unique_ptr<Decl> &decl) -> CherryResult {
  auto loc = tokenLoc();
  consume(Token::kw_struct);

  // Parse Type
  unique_ptr<Type> type;
  if (parseType(type) || parseToken(Token::l_brace, diag::expected_l_brace))
    return failure();

  // Parse Element
  PE<VariableStat> parseField =
      [this](unique_ptr<VariableStat> &elem) -> CherryResult {
    unique_ptr<VariableExpr> var;
    unique_ptr<Type> type;
    if (parseVariableExpr(var) ||
        parseToken(Token::colon, diag::expected_colon) || parseType(type))
      return failure();
    elem = make_unique<VariableStat>(var->location(), std::move(var),
                                     std::move(type), nullptr);
    return success();
  };

  // Parse List
  VectorUniquePtr<VariableStat> fields;
  if (parseList(Token::comma, Token::r_brace, diag::expected_comma_or_r_brace,
                diag::expected_r_brace, fields, parseField))
    return failure();

  // Make StructDecl
  decl = make_unique<StructDecl>(loc, std::move(type), std::move(fields));
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
                              Token::Kind separator, Token::Kind end,
                              const char *const separator_error,
                              const char *const end_error) -> CherryResult {
  PE<Expr> parseElement = [this](unique_ptr<Expr> &elem) -> CherryResult {
    return parseExpression(elem);
  };
  return parseList(separator, end, separator_error, end_error, expressions,
                   parseElement);
}

auto Parser::parsePrimaryExpression(unique_ptr<Expr> &expr) -> CherryResult {
  switch (tokenKind()) {
  case Token::decimal:
    return parseDecimal_c(expr);
  case Token::identifier:
    return parseFuncStructVar_c(expr);
  case Token::kw_if:
    return parseIfExpr_c(expr);
  case Token::kw_while:
    return parseWhileExpr_c(expr);
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
  case Token::l_paren: {
    auto location = tokenLoc();
    consume(Token::l_paren);
    if (parseToken(Token::r_paren, diag::expected_l_paren))
      return failure();
    expr = make_unique<UnitExpr>(location);
    return success();
  }
  default:
    return emitError(diag::expected_expr);
  }
}

auto Parser::parseVariableExpr(unique_ptr<VariableExpr> &identifier)
    -> CherryResult {
  auto location = tokenLoc();
  auto name = spelling();
  if (parseToken(Token::identifier, diag::expected_id))
    return failure();
  identifier = make_unique<VariableExpr>(location, name);
  return success();
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

  expr = make_unique<IfExpr>(loc, std::move(condition), std::move(thenBlock),
                             std::move(elseBlock));
  return success();
}

auto Parser::parseWhileExpr_c(std::unique_ptr<Expr> &expr) -> CherryResult {
  auto loc = tokenLoc();
  consume(Token::kw_while);
  unique_ptr<Expr> condition;
  unique_ptr<BlockExpr> bodyBlock;
  if (parseExpression(condition) ||
      parseToken(Token::l_brace, diag::expected_l_brace) ||
      parseBlockExpr(bodyBlock))
    return failure();
  expr =
      make_unique<WhileExpr>(loc, std::move(condition), std::move(bodyBlock));
  return success();
}

auto Parser::parseDecimal_c(unique_ptr<Expr> &expr) -> CherryResult {
  auto loc = tokenLoc();
  if (auto value = token().getUInt64IntegerValue()) {
    consume(Token::decimal);
    expr = make_unique<DecimalLiteralExpr>(loc, std::move(*value));
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

auto Parser::parseFunctionCall_c(llvm::SMLoc location, llvm::StringRef name,
                                 unique_ptr<Expr> &expr) -> CherryResult {
  consume(Token::l_paren);
  auto expressions = VectorUniquePtr<Expr>();
  if (parseExpressions(expressions, Token::comma, Token::r_paren,
                       diag::expected_comma_or_r_paren, diag::expected_r_paren))
    return failure();
  expr = make_unique<CallExpr>(location, name, std::move(expressions));
  return success();
}

auto Parser::parseBinaryExpRHS(int exprPrec, std::unique_ptr<Expr> &expr)
    -> CherryResult {
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
    if (tokPrec < nextPrec) {
      if (parseBinaryExpRHS(tokPrec + 1, rhs))
        return failure();
    } else if ((tokPrec == nextPrec) && isTokenRightAssociative()) {
      if (parseBinaryExpRHS(tokPrec, rhs))
        return failure();
    }
    expr = std::make_unique<BinaryExpr>(location, tokenToOperator(t),
                                        std::move(expr), std::move(rhs));
  }
}

auto Parser::getTokenPrecedence() -> int {
  switch (tokenKind()) {
  case Token::assign:
    return 100;
  case Token::kw_or:
    return 200;
  case Token::kw_and:
    return 300;
  case Token::kw_eq:
  case Token::kw_neq:
    return 400;
  case Token::kw_lt:
  case Token::kw_le:
  case Token::kw_gt:
  case Token::kw_ge:
    return 500;
  case Token::add:
  case Token::diff:
    return 600;
  case Token::mul:
  case Token::div:
  case Token::rem:
    return 700;
  case Token::dot:
    return 800;
  default:
    return -1;
  }
}

auto Parser::isTokenRightAssociative() -> bool {
  switch (tokenKind()) {
  case Token::assign:
    return true;
  default:
    return false;
  }
}

auto Parser::tokenToOperator(Token token) -> BinaryExpr::Operator {
  switch (token.getKind()) {
  case Token::assign:
    return BinaryExpr::Operator::Assign;
  case Token::dot:
    return BinaryExpr::Operator::StructRead;
  case Token::add:
    return BinaryExpr::Operator::Add;
  case Token::diff:
    return BinaryExpr::Operator::Diff;
  case Token::mul:
    return BinaryExpr::Operator::Mul;
  case Token::div:
    return BinaryExpr::Operator::Div;
  case Token::rem:
    return BinaryExpr::Operator::Rem;
  case Token::kw_and:
    return BinaryExpr::Operator::And;
  case Token::kw_or:
    return BinaryExpr::Operator::Or;
  case Token::kw_eq:
    return BinaryExpr::Operator::EQ;
  case Token::kw_neq:
    return BinaryExpr::Operator::NEQ;
  case Token::kw_lt:
    return BinaryExpr::Operator::LT;
  case Token::kw_le:
    return BinaryExpr::Operator::LE;
  case Token::kw_gt:
    return BinaryExpr::Operator::GT;
  case Token::kw_ge:
    return BinaryExpr::Operator::GE;
  default:
    llvm_unreachable("Unexpected operator");
  }
}

// _____________________________________________________________________________
// Parse Statements

auto Parser::parseStatementWithoutSemi(unique_ptr<Stat> &stat) -> CherryResult {
  switch (tokenKind()) {
  case Token::kw_var:
    return parseVarDecl_c(stat);
  default: {
    auto loc = tokenLoc();
    unique_ptr<Expr> expr;
    if (parseExpression(expr))
      return failure();
    stat = make_unique<ExprStat>(loc, std::move(expr));
    return success();
  }
  }
}

auto Parser::parseVarDecl_c(unique_ptr<Stat> &stat) -> CherryResult {
  auto loc = tokenLoc();
  consume(Token::kw_var);
  unique_ptr<VariableExpr> var;
  unique_ptr<Type> type;
  unique_ptr<Expr> e;
  if (parseVariableExpr(var) ||
      parseToken(Token::colon, diag::expected_colon) || parseType(type) ||
      parseToken(Token::assign, diag::expected_assign) || parseExpression(e))
    return failure();
  stat = make_unique<VariableStat>(loc, std::move(var), std::move(type),
                                   std::move(e));
  return success();
}