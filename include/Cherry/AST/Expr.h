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
  };

  explicit Expr(ExpressionKind kind,
                llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> ExpressionKind { return _kind; }

  virtual auto isLvalue() -> bool { return false; }

private:
  const ExpressionKind _kind;
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
// VariableDecl

class VariableDeclExpr final : public Expr {
public:
  explicit VariableDeclExpr(llvm::SMLoc location,
                            std::unique_ptr<VariableExpr> variable,
                            std::unique_ptr<Identifier> type,
                            std::unique_ptr<Expr> init)
      : Expr{Expr_VariableDecl, location}, _variable(std::move(variable)),
        _type(std::move(type)), _init{std::move(init)} {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_VariableDecl;
  }

  auto variable() const -> const std::unique_ptr<VariableExpr>& {
    return _variable;
  }

  auto type() const -> const std::unique_ptr<Identifier>& {
    return _type;
  }
  auto init() const -> const std::unique_ptr<Expr>& {
    return _init;
  }

private:
  std::unique_ptr<VariableExpr> _variable;
  std::unique_ptr<Identifier> _type;
  std::unique_ptr<Expr> _init;
};


} // end namespace cherry

#endif // CHERRY_EXPR_H
