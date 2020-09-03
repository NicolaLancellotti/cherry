#ifndef CHERRY_PARSER_H
#define CHERRY_PARSER_H

#include "cherry/Parse/Lexer.h"
#include "cherry/AST/AST.h"
#include "cherry/Parse//DiagnosticsParse.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace cherry {

class Parser {
public:
  Parser(std::unique_ptr<Lexer> lexer,
         llvm::SourceMgr& sourceManager)
      : _token{lexer->lexToken()},
        _lexer{std::move(lexer)},
        _sourceManager{sourceManager} {}

  auto parseModule() -> std::unique_ptr<ModuleAST> {
    auto loc = tokenLocation();
    std::vector<DeclarationAST> declarations;
    do {
      auto decl = parseDeclaration();
      if (!decl)
        return nullptr;
      declarations.push_back(std::move(*decl));
    } while (!tokenIs(Token::eof));

    return std::make_unique<ModuleAST>(loc, std::move(declarations));
  }

private:
  Token _token;
  std::unique_ptr<Lexer> _lexer;
  llvm::SourceMgr& _sourceManager;

  // ___________________________________________________________________________
  // Lex

  auto tokenIs(Token::Kind kind) -> bool { return _token.is(kind); }
  auto tokenKind() -> Token::Kind { return _token.getKind(); }
  auto tokenLocation() -> llvm::SMLoc { return _token.getLoc(); }
  auto spelling() -> llvm::StringRef { return _token.getSpelling(); }
  auto consume(Token::Kind kind) -> void {
    assert(_token.is(kind) && "consume Token mismatch expectation");
    _token = _lexer->lexToken();
  }

  // ___________________________________________________________________________
  // Errors

  template <typename R>
  std::unique_ptr<R> parseError(llvm::SMLoc location,
                                const llvm::Twine &msg) {
    _sourceManager.PrintMessage(location,
                                llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return nullptr;
  }

  // ___________________________________________________________________________
  // Parse Declarations

  auto parseDeclaration() -> std::unique_ptr<DeclarationAST> {
    switch (tokenKind()) {
    case Token::kw_fun:
      return parseFunctionDecl();
    default:
      return parseError<DeclarationAST>(tokenLocation(), diag::expected_fun);
    }
  }

  auto parseFunctionDecl() -> std::unique_ptr<FunctionDeclAST> {
    auto loc = tokenLocation();
    auto proto = parsePrototype();
    if (!proto)
      return nullptr;

    auto body = parseFunctionBody();
    if (!body)
      return nullptr;

    return std::make_unique<FunctionDeclAST>(loc, std::move(proto), std::move(*body));
  }

  auto parsePrototype() -> std::unique_ptr<PrototypeAST> {
    auto location = tokenLocation();
    consume(Token::kw_fun);

    if (!tokenIs(Token::identifier))
      return parseError<PrototypeAST>(tokenLocation(),
                                      diag::expected_id_in_func_decl);
    std::string name{spelling()};
    consume(Token::identifier);

    if (!tokenIs(Token::l_paren))
      return parseError<PrototypeAST>(tokenLocation(),
                                      diag::expected_l_paren_in_arg_list);
    consume(Token::l_paren);

    if (!tokenIs(Token::r_paren))
      return parseError<PrototypeAST>(tokenLocation(),
                                      diag::expected_r_paren_in_arg_list);
    consume(Token::r_paren);

    return std::make_unique<PrototypeAST>(location, std::move(name));
  }

  // ___________________________________________________________________________
  // Parse Expressions

  auto parseFunctionBody() -> std::unique_ptr<std::vector<ExpressionAST>> {
    if (!tokenIs(Token::l_brace))
      return parseError<std::vector<ExpressionAST>>(tokenLocation(),
                                                    diag::expected_l_brace_func_body);
    consume(Token::l_brace);

    auto expressions = std::vector<ExpressionAST>();
    while (!tokenIs(Token::r_brace)) {
      auto expr = parseExpression();
      if (!expr)
        return nullptr;

      if (!tokenIs(Token::semi))
        return parseError<std::vector<ExpressionAST>>(tokenLocation(),
                                                      diag::expected_semi);
      consume(Token::semi);
    }

    if (!tokenIs(Token::r_brace))
      return parseError<std::vector<ExpressionAST>>(tokenLocation(),
                                                    diag::expected_r_brace_func_body);
    consume(Token::r_brace);

    return std::make_unique<std::vector<ExpressionAST>>(std::move(expressions));
  }

  auto parseExpression() -> std::unique_ptr<ExpressionAST> {
    switch (tokenKind()) {
    case Token::decimal:
      return parseDecimal();
    case Token::identifier:
      return parseFunctionCall();
    default:
      return parseError<ExpressionAST>(tokenLocation(),diag::expected_expr);
    }
  }

  auto parseDecimal() -> std::unique_ptr<DecimalExprAST> {
    auto loc = tokenLocation();
    if (auto value = _token.getUInt64IntegerValue()) {
      consume(Token::decimal);
      return std::make_unique<DecimalExprAST>(loc, *value);
    }
    return parseError<DecimalExprAST>(tokenLocation(),diag::integer_literal_overflows);
  }

  auto parseFunctionCall() -> std::unique_ptr<FunctionCallExprAST> {
    auto location = tokenLocation();

    std::string name{spelling()};
    consume(Token::identifier);

    if (!tokenIs(Token::l_paren))
      return parseError<FunctionCallExprAST>(tokenLocation(),
                                             diag::expected_l_paren_func_call);
    consume(Token::l_paren);


    auto expressions = std::vector<ExpressionAST>();
    while (!tokenIs(Token::r_paren)) {
      switch (tokenKind()) {
      default:
        return parseError<FunctionCallExprAST>(tokenLocation(),
                                               diag::expected_decimal);
      case Token::decimal:
        auto expr = parseDecimal();
        if (!expr)
          return nullptr;
        expressions.push_back(std::move(*expr));
        break;
      }

      if (tokenIs(Token::r_paren))
        break;

      if (!tokenIs(Token::comma))
        return parseError<FunctionCallExprAST>(tokenLocation(),
                                               diag::expected_comma_or_l_paren_arg_list);
      consume(Token::comma);

    }

    consume(Token::r_paren);
    return std::make_unique<FunctionCallExprAST>(location, std::move(name), std::move(expressions));
  }

};

}

#endif