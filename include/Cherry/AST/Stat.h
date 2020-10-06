//===--- Stat.h - Cherry Language Expression ASTs ---------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_STAT_H
#define CHERRY_STAT_H

#include "Node.h"
#include "llvm/ADT/StringRef.h"

namespace cherry {
class Expr;
class Identifier;
class VariableExpr;

// _____________________________________________________________________________
// Expression

class Stat : public Node {
public:
  enum StatementKind {
    Stat_VariableDecl,
    Stat_Expression,
  };

  explicit Stat(StatementKind kind,
                llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> StatementKind { return _kind; }

private:
  const StatementKind _kind;
};

// _____________________________________________________________________________
// Variable statement

class VariableStat final : public Stat {
public:
  explicit VariableStat(llvm::SMLoc location,
                        std::unique_ptr<VariableExpr> variable,
                        std::unique_ptr<Identifier> varType,
                        std::unique_ptr<Expr> init)
      : Stat{Stat_VariableDecl, location}, _variable(std::move(variable)),
        _varType(std::move(varType)), _init{std::move(init)} {};

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_VariableDecl;
  }

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

// _____________________________________________________________________________
// Expression statement

class ExprStat final : public Stat {
public:
  explicit ExprStat(llvm::SMLoc location,
                    std::unique_ptr<Expr> expression)
      : Stat{Stat_Expression, location},
        _expression{std::move(expression)} {};

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_Expression;
  }

  auto expression() const -> const std::unique_ptr<Expr>& {
    return _expression;
  }

  auto expression() -> std::unique_ptr<Expr>& {
    return _expression;
  }

private:
  std::unique_ptr<Expr> _expression;
};

} // end namespace cherry

#endif // CHERRY_STAT_H
