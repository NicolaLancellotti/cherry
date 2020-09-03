#ifndef CHERRY_EXPRESSIONSAST_H
#define CHERRY_EXPRESSIONSAST_H

#include "cherry/AST/NodeAST.h"
#include <string>

namespace cherry {

// _____________________________________________________________________________
// ExpressionsAST

class ExpressionAST : public NodeAST {
public:
  explicit ExpressionAST(llvm::SMLoc location) : NodeAST{location} {};
};

// _____________________________________________________________________________
// FunctionCallExprAST

class FunctionCallExprAST : public ExpressionAST {
public:
  explicit FunctionCallExprAST(llvm::SMLoc location,
                               std::string name,
                               std::vector<ExpressionAST> expressions)
      : ExpressionAST{location},
        _name(std::move(name)),
        _expressions(std::move(expressions)) {};

  auto name() const -> const std::string& {
    return _name;
  }

  auto expressions() const -> const std::vector<ExpressionAST>& {
    return _expressions;
  }

private:
  std::string _name;
  std::vector<ExpressionAST> _expressions;
};

// _____________________________________________________________________________
// FunctionCallExprAST

class DecimalExprAST : public ExpressionAST {
public:
  explicit DecimalExprAST(llvm::SMLoc location, uint64_t value)
      : ExpressionAST{location}, _value(value) {};

  auto value() const -> uint64_t {
    return _value;
  }

private:
  uint64_t _value;
};

}

#endif