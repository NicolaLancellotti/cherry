//===--- Symbols.h - Symbol Table -------------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_SYMBOLS_H
#define CHERRY_SYMBOLS_H

#include "cherry/AST/AST.h"
#include "cherry/Basic/CherryResult.h"
#include <map>

namespace cherry {
using mlir::failure;
using mlir::success;

class Symbols {
public:
  auto addBuiltins() -> void {
    _typeSymbols.insert(make_pair(UInt64Type,
                                  llvm::SmallVector<std::string, 2>{}));
    _functionSymbols.insert(make_pair("print",
                                      llvm::SmallVector<std::string, 2>{UInt64Type}));
  }

  auto declareFunction(std::string name, llvm::SmallVector<std::string, 2> types) -> CherryResult {
    if (_functionSymbols.find(name) != _functionSymbols.end())
      return failure();
    _functionSymbols.insert(make_pair(name, std::move(types)));
    return success();
  }

  auto getFunction(std::string name,
                   llvm::ArrayRef<std::string>& types)  -> CherryResult {
    auto symbol = _functionSymbols.find(name);
    if (symbol == _functionSymbols.end())
      return failure();
    types = symbol->second;
    return success();
  }

  auto declareType(const Identifier *node,
                   llvm::SmallVector<std::string, 2> types) -> CherryResult {
    auto name = node->name();
    if (_typeSymbols.find(name) != _typeSymbols.end())
      return failure();
    _typeSymbols.insert(make_pair(name, std::move(types)));
    return success();
  }

  auto checkType(const std::string& type) -> CherryResult {
    if (_typeSymbols.find(type) == _typeSymbols.end())
      return failure();
    return success();
  }

  auto getType(std::string name, llvm::ArrayRef<std::string>& types) -> CherryResult {
    auto symbol = _typeSymbols.find(name);
    if (symbol == _typeSymbols.end())
      return failure();
    types = symbol->second;
    return success();
  }

  auto resetVariables() {
    _variableSymbols = {};
  }

  auto declareVariable(const VariableExpr *var, std::string type) -> CherryResult {
    auto name = var->name();
    if (_variableSymbols.find(name) != _variableSymbols.end())
      return failure();
    _variableSymbols.insert(make_pair(name, type));
    return success();
  }

  auto getVariableType(const VariableExpr *node, std::string &type) -> CherryResult {
    auto symbol = _variableSymbols.find(node->name());
    if (symbol == _variableSymbols.end()) {
      return failure();
    }
    type = symbol->second;
    return success();
  }

  const std::string UInt64Type = "UInt64";
private:
  std::map</*name*/ std::string, /*types*/ llvm::SmallVector<std::string, 2>> _functionSymbols;
  std::map</*name*/ std::string, /*types*/ llvm::SmallVector<std::string, 2>> _typeSymbols;
  std::map</*name*/std::string, /*type*/std::string> _variableSymbols;
};

} // end namespace cherry

#endif // CHERRY_SYMBOLS_H
