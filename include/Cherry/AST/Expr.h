//===--- Expr.h - Cherry Language Expression ASTs ---------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_EXPR_H
#define CHERRY_EXPR_H

#include "Node.h"
#include "llvm/ADT/StringRef.h"

namespace cherry {

// _____________________________________________________________________________
// Expression

class Expr : public Node {
public:
  enum ExpressionKind {
    Expr_Call,
    Expr_DecimalLiteral,
    Expr_BoolLiteral,
    Expr_Variable,
    Expr_Struct,
    Expr_Binary,
    Expr_VariableDecl,
    Expr_Block,
    Expr_If,
  };

  explicit Expr(ExpressionKind kind,
                llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> ExpressionKind { return _kind; }

  virtual auto isLvalue() -> bool { return false; }
  virtual auto isStatement() -> bool { return false; };

  virtual auto type() const -> llvm::StringRef final {
    return _type == "" ? llvm::StringRef("NULL") : _type;
  }

  auto setType(llvm::StringRef type) -> void {
    _type = type.str();
  }

private:
  const ExpressionKind _kind;
  std::string _type;
};

// _____________________________________________________________________________
// Call expression

class VariableExpr final : public Expr {
public:
  explicit VariableExpr(llvm::SMLoc location,
                        llvm::StringRef name)
      : Expr{Expr_Variable, location}, _name(name.str()) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Variable;
  }

  auto name() const -> llvm::StringRef {
    return _name;
  }

  auto isLvalue() -> bool {
    return true;
  }

private:
  std::string _name;
};

// _____________________________________________________________________________
// Call expression

class CallExpr final : public Expr {
public:
  explicit CallExpr(llvm::SMLoc location,
                    llvm::StringRef name,
                    VectorUniquePtr<Expr> expressions)
      : Expr{Expr_Call, location},
        _name(name.str()),
        _expressions(std::move(expressions)) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Call;
  }

  auto name() const -> llvm::StringRef {
    return _name;
  }

  auto expressions() const -> const VectorUniquePtr<Expr>& {
    return _expressions;
  }

private:
  std::string _name;
  VectorUniquePtr<Expr> _expressions;

public:
  auto begin() const -> decltype(_expressions.begin()) { return _expressions.begin(); }
  auto end() const -> decltype(_expressions.end()) { return _expressions.end(); }
};

// _____________________________________________________________________________
// Decimal expression

class DecimalLiteralExpr final : public Expr {
public:
  explicit DecimalLiteralExpr(llvm::SMLoc location, uint64_t value)
      : Expr{Expr_DecimalLiteral, location}, _value(value) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_DecimalLiteral;
  }

  auto value() const -> uint64_t {
    return _value;
  }

private:
  uint64_t _value;
};

// _____________________________________________________________________________
// Boolean expression

class BoolLiteralExpr final : public Expr {
public:
  explicit BoolLiteralExpr(llvm::SMLoc location, bool value)
      : Expr{Expr_BoolLiteral, location}, _value(value) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_BoolLiteral;
  }

  auto value() const -> bool {
    return _value;
  }

private:
  bool _value;
};

// _____________________________________________________________________________
// Binary expression

class BinaryExpr final : public Expr {
public:
  explicit BinaryExpr(llvm::SMLoc location,
                      llvm::StringRef op,
                      std::unique_ptr<Expr> lhs,
                      std::unique_ptr<Expr> rhs)
      : Expr{Expr_Binary, location}, _op{op.str()},
        _lhs{std::move(lhs)}, _rhs{std::move(rhs)} {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Binary;
  }

  auto lhs() const -> const std::unique_ptr<Expr>& {
    return _lhs;
  }

  auto rhs() const -> const std::unique_ptr<Expr>& {
    return _rhs;
  }

  auto op() const -> llvm::StringRef {
    return _op;
  }

  auto isLvalue() -> bool {
    return _op == ".";
  }

private:
  std::unique_ptr<Expr> _lhs;
  std::unique_ptr<Expr> _rhs;
  std::string _op;
};

// _____________________________________________________________________________
// Block expression

class BlockExpr final : public Expr {
public:
  explicit BlockExpr(llvm::SMLoc location,
                     VectorUniquePtr<Expr> statements,
                     std::unique_ptr<Expr> expression)
      : Expr{Expr_Block, location},
        _statements(std::move(statements)),
        _expression(std::move(expression)) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Block;
  }

  auto statements() const -> const VectorUniquePtr<Expr>& {
    return _statements;
  }

  auto expression() const -> const std::unique_ptr<Expr>& {
    return _expression;
  }

private:
  VectorUniquePtr<Expr> _statements;
  std::unique_ptr<Expr> _expression;
public:
  auto begin() const -> decltype(_statements.begin()) { return _statements.begin(); }
  auto end() const -> decltype(_statements.end()) { return _statements.end(); }
};

// _____________________________________________________________________________
// If expression

class IfExpr final : public Expr {
public:
  explicit IfExpr(llvm::SMLoc location,
                  std::unique_ptr<Expr> condition,
                  std::unique_ptr<BlockExpr> thenExpr,
                  std::unique_ptr<BlockExpr>  elseExpr)
      : Expr{Expr_If, location}, _condition(std::move(condition)),
        _thenExpr(std::move(thenExpr)), _elseExpr(std::move(elseExpr)) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_If;
  }

  auto conditionExpr() const -> const std::unique_ptr<Expr>& {
    return _condition;
  }

  auto thenBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _thenExpr;
  }

  auto elseBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _elseExpr;
  }

private:
  std::unique_ptr<Expr> _condition;
  std::unique_ptr<BlockExpr>  _thenExpr;
  std::unique_ptr<BlockExpr>  _elseExpr;
};

// _____________________________________________________________________________
// Variable declaration

class VariableDeclExpr final : public Expr {
public:
  explicit VariableDeclExpr(llvm::SMLoc location,
                            std::unique_ptr<VariableExpr> variable,
                            std::unique_ptr<Identifier> varType,
                            std::unique_ptr<Expr> init)
      : Expr{Expr_VariableDecl, location}, _variable(std::move(variable)),
        _varType(std::move(varType)), _init{std::move(init)} {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_VariableDecl;
  }

  auto isStatement() -> bool override {
    return true;
  };

  auto variable() const -> const std::unique_ptr<VariableExpr>& {
    return _variable;
  }

  auto varType() const -> const std::unique_ptr<Identifier>& {
    return _varType;
  }
  auto init() const -> const std::unique_ptr<Expr>& {
    return _init;
  }

private:
  std::unique_ptr<VariableExpr> _variable;
  std::unique_ptr<Identifier> _varType;
  std::unique_ptr<Expr> _init;
};

} // end namespace cherry

#endif // CHERRY_EXPR_H
