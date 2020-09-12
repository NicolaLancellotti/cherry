//===--- Expr.h - Cherry Language Expression ASTs ---------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_EXPR_H
#define CHERRY_EXPR_H

#include "Node.h"
#include <string>
#include <vector>

namespace cherry {

// _____________________________________________________________________________
// Expression

class Expr : public Node {
public:
  enum ExpressionKind {
    Expr_Call,
    Expr_Decimal,
    Expr_Variable,
  };

  explicit Expr(ExpressionKind kind,
                      llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> ExpressionKind { return _kind; }

private:
  const ExpressionKind _kind;
};

// _____________________________________________________________________________
// Call expression

class Variable final : public Expr {
public:
  explicit Variable(llvm::SMLoc location,
                      std::string name)
      : Expr{Expr_Variable, location}, _name(std::move(name)) {};

  static auto classof(const Expr * node) -> bool {
    return node->getKind() == Expr_Variable;
  }

  auto name() const -> const std::string& {
    return _name;
  }

private:
  std::string _name;
};

// _____________________________________________________________________________
// Call expression

class CallExpr final : public Expr {
public:
  explicit CallExpr(llvm::SMLoc location,
                    std::string name,
                    VectorUniquePtr<Expr> expressions)
      : Expr{Expr_Call, location},
        _name(std::move(name)),
        _expressions(std::move(expressions)) {};

  static auto classof(const Expr * node) -> bool {
    return node->getKind() == Expr_Call;
  }

  auto name() const -> const std::string& {
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

class DecimalExpr final : public Expr {
public:
  explicit DecimalExpr(llvm::SMLoc location, uint64_t value)
      : Expr{Expr_Decimal, location}, _value(value) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Decimal;
  }

  auto value() const -> uint64_t {
    return _value;
  }

private:
  uint64_t _value;
};

} // end namespace cherry

#endif // CHERRY_EXPR_H
