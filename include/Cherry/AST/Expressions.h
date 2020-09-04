#ifndef CHERRY_EXPRESSIONS_H
#define CHERRY_EXPRESSIONS_H

#include "cherry/AST/Node.h"
#include <string>
#include <vector>

namespace cherry {

// _____________________________________________________________________________
// Expressions

class Expr : public Node {
public:
  enum ExpressionKind {
    Expr_Call,
    Expr_Decimal,
  };

  explicit Expr(ExpressionKind kind,
                      llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> ExpressionKind { return _kind; }

private:
  const ExpressionKind _kind;
};

// _____________________________________________________________________________
// CallExpr

class CallExpr : public Expr {
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
// DecimalExpr

class DecimalExpr : public Expr {
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

}

#endif