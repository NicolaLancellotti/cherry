#ifndef CHERRY_PARSER_H
#define CHERRY_PARSER_H

#include "cherry/AST/AST.h"
#include "cherry/Parse//DiagnosticsParse.h"
#include "cherry/Parse/Lexer.h"
#include "cherry/Parse/ParseResult.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace cherry {

using mlir::failure;
using mlir::success;

class Parser {
public:
  Parser(std::unique_ptr<Lexer> lexer,
         llvm::SourceMgr& sourceManager)
      : _token{lexer->lexToken()},
        _lexer{std::move(lexer)},
        _sourceManager{sourceManager} {}

  auto parseModule(std::unique_ptr<Module>& module) -> ParseResult  {
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

  ParseResult emitError(const llvm::Twine &msg) {
    _sourceManager.PrintMessage(tokenLoc(),
                                llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
  }

  // ___________________________________________________________________________
  // Parse Token

  auto parseToken(Token::Kind expected,
                  const llvm::Twine &message) -> ParseResult {
    if (consumeIf(expected))
      return success();
    return emitError(message);
  }

  // ___________________________________________________________________________
  // Parse Declarations

  auto parseDeclaration(std::unique_ptr<Decl>& decl) -> ParseResult {
    switch (tokenKind()) {
    case Token::kw_fun: {
      std::unique_ptr<Decl> func;
      if (parseFunctionDecl_c(func))
        return failure();
      decl = std::move(func);
      return success();
    }
    default:
      return emitError(diag::expected_fun);
    }
  }

  auto parseFunctionDecl_c(std::unique_ptr<Decl>& decl) -> ParseResult {
    auto loc = tokenLoc();
    std::unique_ptr<Prototype> proto;
    VectorUniquePtr<Expr> body;

    if (parsePrototype_c(proto) || parseFunctionBody(body))
      return failure();

    decl = std::make_unique<FunctionDecl>(loc, std::move(proto), std::move(body));

    return success();
  }

  auto parsePrototype_c(std::unique_ptr<Prototype>& proto) -> ParseResult {
    auto location = tokenLoc();
    consume(Token::kw_fun);

    std::string name{spelling()};
    if (parseToken(Token::identifier,
                   diag::expected_id_in_func_decl) ||
        parseToken(Token::l_paren,
                   diag::expected_l_paren_in_arg_list))
      return failure();

    if (parseToken(Token::r_paren,
                   diag::expected_r_paren_in_arg_list))
      return failure();

    proto = std::make_unique<Prototype>(location, std::move(name));
    return success();
  }

  // ___________________________________________________________________________
  // Parse Expressions

  auto parseFunctionBody(VectorUniquePtr<Expr>& expressions) -> ParseResult {
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

  auto parseExpression(std::unique_ptr<Expr>& expr) -> ParseResult {
    switch (tokenKind()) {
    case Token::decimal: {
      std::unique_ptr<DecimalExpr> decimal;
      if (parseDecimal_c(decimal))
        return failure();
      expr = std::move(decimal);
      return success();
    }
    case Token::identifier: {
      std::unique_ptr<CallExpr> fun;
      if (parseFunctionCall_c(fun))
        return failure();
      expr = std::move(fun);
      return success();
    }
    default:
      return emitError(diag::expected_expr);
    }
  }

  auto parseDecimal_c(std::unique_ptr<DecimalExpr>& expr) -> ParseResult {
    auto loc = tokenLoc();
    if (auto value = token().getUInt64IntegerValue()) {
      consume(Token::decimal);
      expr = std::make_unique<DecimalExpr>(loc, std::move(*value));
      return success();
    }
    return emitError(diag::integer_literal_overflows);
  }

  auto parseFunctionCall_c(std::unique_ptr<CallExpr>& expr) -> ParseResult {
    auto location = tokenLoc();
    std::string name{spelling()};
    consume(Token::identifier);

    if (parseToken(Token::l_paren, diag::expected_l_paren_func_call))
      return failure();

    auto expressions = VectorUniquePtr<Expr>();
    while (!tokenIs(Token::r_paren)) {
      switch (tokenKind()) {
      case Token::decimal: {
        std::unique_ptr<DecimalExpr> decimal;
        if (parseDecimal_c(decimal))
          return failure();
        expressions.push_back(std::move(decimal));
        break;
      }
      default:
        return emitError(diag::expected_decimal);
      }

      if (tokenIs(Token::r_paren))
        break;

      if (parseToken(Token::comma,
                     diag::expected_comma_or_l_paren_arg_list))
        return failure();
    }

    consume(Token::r_paren);
    expr = std::make_unique<CallExpr>(location, std::move(name),
                                              std::move(expressions));
    return success();
  }

};

}

#endif