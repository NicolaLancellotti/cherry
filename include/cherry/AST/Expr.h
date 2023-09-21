//===--- Expr.h - Cherry Language Expression ASTs ---------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_EXPR_H
#define CHERRY_EXPR_H

#include "cherry/AST/Node.h"
#include "llvm/ADT/StringRef.h"

namespace cherry {
class Stat;
// _____________________________________________________________________________
// Expression

class Expr : public Node {
public:
  enum ExpressionKind {
    Expr_Unit,
    Expr_Call,
    Expr_DecimalLiteral,
    Expr_BoolLiteral,
    Expr_Variable,
    Expr_Struct,
    Expr_Binary,
    Expr_Block,
    Expr_If,
    Expr_While,
  };

  explicit Expr(ExpressionKind kind, llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> ExpressionKind { return _kind; }

  virtual auto isLvalue() -> bool { return false; }
  virtual auto isStatement() -> bool { return false; };

  virtual auto type() const -> llvm::StringRef final {
    return _type == "" ? llvm::StringRef("NULL") : _type;
  }

  auto setType(llvm::StringRef type) -> void { _type = type.str(); }

private:
  const ExpressionKind _kind;
  std::string _type;
};

// _____________________________________________________________________________
// Unit

class UnitExpr final : public Expr {
public:
  explicit UnitExpr(llvm::SMLoc location) : Expr{Expr_Unit, location} {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Unit;
  }
};

// _____________________________________________________________________________
// Call expression

class VariableExpr final : public Expr {
public:
  explicit VariableExpr(llvm::SMLoc location, llvm::StringRef name)
      : Expr{Expr_Variable, location}, _name(name.str()){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Variable;
  }

  auto name() const -> llvm::StringRef { return _name; }

  auto isLvalue() -> bool override { return true; }

private:
  std::string _name;
};

// _____________________________________________________________________________
// Call expression

class CallExpr final : public Expr {
public:
  explicit CallExpr(llvm::SMLoc location, llvm::StringRef name,
                    VectorUniquePtr<Expr> expressions)
      : Expr{Expr_Call, location}, _name(name.str()),
        _expressions(std::move(expressions)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Call;
  }

  auto name() const -> llvm::StringRef { return _name; }

  auto expressions() const -> const VectorUniquePtr<Expr> & {
    return _expressions;
  }

private:
  std::string _name;
  VectorUniquePtr<Expr> _expressions;

public:
  auto begin() const -> decltype(_expressions.begin()) {
    return _expressions.begin();
  }
  auto end() const -> decltype(_expressions.end()) {
    return _expressions.end();
  }
};

// _____________________________________________________________________________
// Decimal expression

class DecimalLiteralExpr final : public Expr {
public:
  explicit DecimalLiteralExpr(llvm::SMLoc location, uint64_t value)
      : Expr{Expr_DecimalLiteral, location}, _value(value){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_DecimalLiteral;
  }

  auto value() const -> uint64_t { return _value; }

private:
  uint64_t _value;
};

// _____________________________________________________________________________
// Boolean expression

class BoolLiteralExpr final : public Expr {
public:
  explicit BoolLiteralExpr(llvm::SMLoc location, bool value)
      : Expr{Expr_BoolLiteral, location}, _value(value){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_BoolLiteral;
  }

  auto value() const -> bool { return _value; }

private:
  bool _value;
};

// _____________________________________________________________________________
// Binary expression

class BinaryExpr final : public Expr {
public:
  enum class Operator {
    Assign,
    StructRead,
    Add,
    Mul,
    Diff,
    Div,
    Rem,
    And,
    Or,
    LT,
    LE,
    GT,
    GE,
    EQ,
    NEQ,
  };

  explicit BinaryExpr(llvm::SMLoc location, BinaryExpr::Operator op,
                      std::unique_ptr<Expr> lhs, std::unique_ptr<Expr> rhs)
      : Expr{Expr_Binary, location}, _op{op}, _lhs{std::move(lhs)},
        _rhs{std::move(rhs)} {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Binary;
  }

  auto lhs() const -> const std::unique_ptr<Expr> & { return _lhs; }

  auto rhs() const -> const std::unique_ptr<Expr> & { return _rhs; }

  auto op() const -> llvm::StringRef {
    switch (_op) {
    case Operator::Assign:
      return "=";
    case Operator::StructRead:
      return ".";
    case Operator::Add:
      return "+";
    case Operator::Diff:
      return "-";
    case Operator::Mul:
      return "*";
    case Operator::Div:
      return "/";
    case Operator::Rem:
      return "%";
    case Operator::And:
      return "and";
    case Operator::Or:
      return "or";
    case Operator::LT:
      return "lt";
    case Operator::LE:
      return "le";
    case Operator::GT:
      return "gt";
    case Operator::GE:
      return "ge";
    case Operator::EQ:
      return "eq";
    case Operator::NEQ:
      return "neq";
    }
  }

  auto opEnum() const -> Operator { return _op; }

  auto isLvalue() -> bool override { return _op == Operator::StructRead; }

  auto index() const -> int { return _index; }

  auto setIndex(int index) { _index = index; }

private:
  Operator _op;
  std::unique_ptr<Expr> _lhs;
  std::unique_ptr<Expr> _rhs;
  int _index;
};

// _____________________________________________________________________________
// Block expression

class BlockExpr final : public Expr {
public:
  explicit BlockExpr(llvm::SMLoc location, VectorUniquePtr<Stat> statements,
                     std::unique_ptr<Expr> expression)
      : Expr{Expr_Block, location}, _statements(std::move(statements)),
        _expression(std::move(expression)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Block;
  }

  auto statements() const -> const VectorUniquePtr<Stat> & {
    return _statements;
  }

  auto expression() const -> const std::unique_ptr<Expr> & {
    return _expression;
  }

private:
  VectorUniquePtr<Stat> _statements;
  std::unique_ptr<Expr> _expression;

public:
  auto begin() const -> decltype(_statements.begin()) {
    return _statements.begin();
  }
  auto end() const -> decltype(_statements.end()) { return _statements.end(); }
};

// _____________________________________________________________________________
// If expression

class IfExpr final : public Expr {
public:
  explicit IfExpr(llvm::SMLoc location, std::unique_ptr<Expr> condition,
                  std::unique_ptr<BlockExpr> thenExpr,
                  std::unique_ptr<BlockExpr> elseExpr)
      : Expr{Expr_If, location}, _condition(std::move(condition)),
        _thenExpr(std::move(thenExpr)), _elseExpr(std::move(elseExpr)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_If;
  }

  auto conditionExpr() const -> const std::unique_ptr<Expr> & {
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
  std::unique_ptr<BlockExpr> _thenExpr;
  std::unique_ptr<BlockExpr> _elseExpr;
};

// _____________________________________________________________________________
// While expression

class WhileExpr final : public Expr {
public:
  explicit WhileExpr(llvm::SMLoc location, std::unique_ptr<Expr> condition,
                     std::unique_ptr<BlockExpr> bodyBlock)
      : Expr{Expr_While, location}, _condition(std::move(condition)),
        _bodyBlock(std::move(bodyBlock)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_While;
  }

  auto conditionExpr() const -> const std::unique_ptr<Expr> & {
    return _condition;
  }

  auto bodyBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _bodyBlock;
  }

private:
  std::unique_ptr<Expr> _condition;
  std::unique_ptr<BlockExpr> _bodyBlock;
};

} // end namespace cherry

#endif // CHERRY_EXPR_H
