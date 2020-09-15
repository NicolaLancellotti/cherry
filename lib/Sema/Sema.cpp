//===--- Sema.cpp - Cherry Semantic Analysis ------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "DiagnosticsSema.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/CherryResult.h"
#include <map>
#include <set>

namespace {
using namespace cherry;
using llvm::cast;
using std::make_pair;
using mlir::failure;
using mlir::success;

class SemaImpl {
public:
  SemaImpl(const llvm::SourceMgr &sourceManager)
      : _sourceManager{sourceManager} {
    addBuiltins();
  }

  auto sema(const Module &node) -> CherryResult {
    for (auto &decl : node) {
      if (sema(decl.get()))
        return failure();
    }

    auto symbol = _functionSymbols.find("main");
    if (symbol == _functionSymbols.end() || symbol->second.size() != 0) {
      return emitError(llvm::SMLoc{}, diag::main_undefined);
    }

    return success();
  }

private:
  const std::string UInt64Type = "UInt64";
  const llvm::SourceMgr &_sourceManager;
  std::map</*name*/ std::string, /*types*/ std::vector<std::string>> _functionSymbols;
  std::set<std::string> _typeSymbols;
  std::map</*name*/std::string, /*type*/std::string> _variableSymbols;

  // Symbols

  auto addBuiltins() -> void {
    _typeSymbols.insert(UInt64Type);
    _functionSymbols.insert(make_pair("print",
                                      std::vector<std::string>{UInt64Type}));
  }

  auto declareFunction(std::string name, std::vector<std::string> types) -> CherryResult {
    if (getFunction(name))
      return failure();
    _functionSymbols.insert(make_pair(name, std::move(types)));
    return success();
  }

  auto getFunction(std::string name) -> std::vector<std::string>* {
    auto types = _functionSymbols.find(name);
    if (types == _functionSymbols.end())
      return nullptr;

    return &_functionSymbols[name];
  }

  auto declareType(const Identifier *node) -> CherryResult {
    auto name = node->name();
    if (_typeSymbols.find(name) != _typeSymbols.end())
      return failure();
    _typeSymbols.insert(name);
    return success();
  }

  auto checkTypeExist(const Identifier *node) -> CherryResult {
    if (_typeSymbols.find(node->name()) == _typeSymbols.end())
      return failure();
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

  auto checkVariableExist(const VariableExpr *node) -> CherryResult {
    if (_variableSymbols.find(node->name()) == _variableSymbols.end())
      return failure();
    return success();
  }

  auto variableType(const VariableExpr *node) -> std::string {
    auto symbol = _variableSymbols.find(node->name());
    if (symbol == _variableSymbols.end()) {
      return "";
    }
    return symbol->second;
  }

  // Errors

  auto emitError(const Node *node, const llvm::Twine &msg) -> CherryResult {
    _sourceManager.PrintMessage(node->location(),
                                llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
  }

  auto emitError(llvm::SMLoc loc, const llvm::Twine &msg) -> CherryResult {
    _sourceManager.PrintMessage(loc,
                                llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
  }

  // Semantic Analysis

  auto sema(const Decl *node) -> CherryResult {
    switch (node->getKind()) {
    case Decl::Decl_Function: {
      return sema(cast<FunctionDecl>(node));
    }
    case Decl::Decl_Struct: {
      return sema(cast<StructDecl>(node));
    }
    default:
      return failure();
    }
  }

  auto sema(const FunctionDecl *node) -> CherryResult {
    resetVariables();
    if (sema(node->proto().get()))
      return failure();

    for (auto &expr : *node) {
      std::string type;
      if (sema(expr.get(), type))
        return failure();
    }
    return success();
  }

  auto sema(const Prototype *node) -> CherryResult {
    std::vector<std::string> types;
    for (auto &par : node->parameters()) {
      auto type = par->type().get();
      auto typeName = type->name();
      if (checkTypeExist(type))
        return emitError(type, diag::type_undefined);
      if (declareVariable(par->variable().get(), typeName))
        return emitError(par->variable().get(), diag::var_redefinition);
      types.push_back(typeName);
    }

    auto name = node->id()->name();
    if (declareFunction(name, std::move(types))) {
      const char *diagnostic = diag::func_redefinition;
      char buffer[50];
      sprintf(buffer, diagnostic, name.c_str());
      return emitError(node->id().get(), buffer);
    }
    return success();
  }

  auto sema(const StructDecl *node) -> CherryResult {
    std::set<std::string> variables;
    for (auto &varDecl : *node) {
      auto type = varDecl->type().get();
      auto var = varDecl->variable().get();
      if (checkTypeExist(type))
        return emitError(type, diag::type_undefined);
      if (variables.find(var->name()) != variables.end())
        return emitError(var, diag::var_redefinition);
      variables.insert(var->name());
    }
    auto id = node->id().get();
    if (declareType(id))
      return emitError(id, diag::type_redefinition);
    return success();
  }

  auto sema(const Expr *node, std::string& type) -> CherryResult {
    switch (node->getKind()) {
    case Expr::Expr_Decimal:
      return sema(cast<DecimalExpr>(node), type);
    case Expr::Expr_Call:
      return sema(cast<CallExpr>(node), type);
    case Expr::Expr_Variable:
      return sema(cast<VariableExpr>(node), type);
    default:
      return failure();
    }
  }

  auto sema(const VariableExpr *node, std::string& type) -> CherryResult {
    if (checkVariableExist(node))
      return emitError(node, diag::var_undefined);
    type = variableType(node);
    return success();
  }

  auto sema(const CallExpr *node, std::string& type) -> CherryResult {
    auto name = node->name();
    std::vector<std::string>* parametersTypes = getFunction(name);
    if (!parametersTypes) {
      const char * diagnostic = diag::func_undefined;
      char buffer [50];
      sprintf(buffer, diagnostic, name.c_str());
      return emitError(node, buffer);
    }

    auto &expressions = node->expressions();
    if (expressions.size() != parametersTypes->size()) {
      const char * diagnostic = diag::func_param;
      char buffer [50];
      sprintf(buffer, diagnostic, name.c_str(), parametersTypes->size());
      return emitError(node, buffer);
    }

    for (int i = 0; i < expressions.size(); ++i) {
      auto type = (*parametersTypes)[i];
      auto &expr = expressions[i];
      std::string exprType;
      if (sema(expr.get(), exprType))
        return failure();
      if (exprType != type)
        return emitError(expr.get(), diag::func_param_type_mismatch);
    }
    type = UInt64Type;
    return success();
  }

  auto sema(const DecimalExpr *node, std::string& type) -> CherryResult {
    type = UInt64Type;
    return success();
  }
};

} // end namespace

namespace cherry {

auto sema(const llvm::SourceMgr &sourceManager,
          const Module &moduleAST) -> CherryResult {
  return SemaImpl(sourceManager).sema(moduleAST);
}

} // end namespace cherry
