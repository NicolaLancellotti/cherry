//===--- Symbols.h - Symbol Table -------------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_SYMBOLS_H
#define CHERRY_SYMBOLS_H

#include "cherry/AST/AST.h"
#include "cherry/Basic/Builtins.h"
#include "cherry/Basic/CherryResult.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace cherry {
using mlir::failure;
using mlir::success;

class Symbols {
public:
  auto addBuiltins() -> void {
    for (auto type : builtins::primitiveTypes())
      _typeSymbols.insert(std::make_pair(type, &emptyVector));

    declareFunction(builtins::print,
                    llvm::SmallVector<llvm::StringRef, 1>{builtins::UInt64Type},
                    builtins::UInt64Type);

    declareFunction(builtins::boolToUInt64,
                    llvm::SmallVector<llvm::StringRef, 1>{builtins::BoolType},
                    builtins::UInt64Type);
  }

  auto declareFunction(llvm::StringRef name,
                       llvm::SmallVector<llvm::StringRef, 2> types,
                       llvm::StringRef returnType) -> CherryResult {
    if (_functionSymbols.find(name) != _functionSymbols.end())
      return failure();
    auto value = std::make_pair(std::move(types), returnType);
    _functionSymbols.insert(std::make_pair(name, std::move(value)));
    return success();
  }

  auto getFunction(llvm::StringRef name,
                   llvm::ArrayRef<llvm::StringRef> &types,
                   llvm::StringRef &returnType)  -> CherryResult {
    auto symbol = _functionSymbols.find(name);
    if (symbol == _functionSymbols.end())
      return failure();

    types = symbol->second.first;
    returnType = symbol->second.second;
    return success();
  }

  auto declareType(const StructDecl *node) -> CherryResult {
    auto name = node->id()->name();
    if (_typeSymbols.find(name.str()) != _typeSymbols.end())
      return failure();

    _typeSymbols.insert(std::make_pair(name, &(node->variables())));
    return success();
  }

  auto checkType(llvm::StringRef type) -> CherryResult {
    if (_typeSymbols.find(type) == _typeSymbols.end())
      return failure();
    return success();
  }

  auto getType(llvm::StringRef name,
               const VectorUniquePtr<VariableStat> *&types) -> CherryResult {
    auto symbol = _typeSymbols.find(name);
    if (symbol == _typeSymbols.end())
      return failure();
    types = symbol->second;
    return success();
  }

  auto resetVariables() {
    _variableSymbols = {};
  }

  auto declareVariable(const VariableExpr *var, llvm::StringRef type) -> CherryResult {
    auto name = var->name();
    if (_variableSymbols.find(name) != _variableSymbols.end())
      return failure();
    _variableSymbols.insert(std::make_pair(name, type));
    return success();
  }

  auto getVariableType(const VariableExpr *node, llvm::StringRef &type) -> CherryResult {
    auto symbol = _variableSymbols.find(node->name());
    if (symbol == _variableSymbols.end()) {
      return failure();
    }
    type = symbol->second;
    return success();
  }

  VectorUniquePtr<VariableStat> emptyVector;
private:
  std::map</*name*/ llvm::StringRef,
      std::pair<
          /*_functionSymbolstypes*/ llvm::SmallVector<llvm::StringRef, 2>,
          /*return type*/ llvm::StringRef>> _functionSymbols;
  std::map</*name*/ llvm::StringRef, /*types*/ const VectorUniquePtr<VariableStat>*> _typeSymbols;
  std::map</*name*/llvm::StringRef, /*type*/ llvm::StringRef> _variableSymbols;
};

} // end namespace cherry

#endif // CHERRY_SYMBOLS_H
