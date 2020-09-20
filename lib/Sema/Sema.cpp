//===--- Sema.cpp - Cherry Semantic Analysis ------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "Symbols.h"
#include "DiagnosticsSema.h"
#include "cherry/AST/AST.h"
#include "llvm/ADT/SmallSet.h"

namespace {
using namespace cherry;
using llvm::cast;

class SemaImpl {
public:
  SemaImpl(const llvm::SourceMgr &sourceManager)
      : _sourceManager{sourceManager} {
    symbols.addBuiltins();
  }

  auto sema(const Module &node) -> CherryResult {
    for (auto &decl : node)
      if (sema(decl.get()))
        return failure();

    llvm::ArrayRef<std::string> types;
    if (symbols.getFunction("main", types) || types.size() != 0)
      return emitError(llvm::SMLoc{}, diag::main_undefined);
    return success();
  }

private:
  const llvm::SourceMgr &_sourceManager;
  Symbols symbols;

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
      llvm_unreachable("Unexpected declaration");
    }
  }

  auto sema(const FunctionDecl *node) -> CherryResult {
    symbols.resetVariables();
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
    llvm::SmallVector<std::string, 2> types;
    for (auto &par : node->parameters()) {
      auto type = par->type().get();
      auto typeName = type->name();
      if (symbols.checkType(type->name()))
        return emitError(type, diag::type_undefined);
      if (symbols.declareVariable(par->variable().get(), typeName))
        return emitError(par->variable().get(), diag::var_redefinition);
      types.push_back(typeName);
    }

    auto name = node->id()->name();
    if (symbols.declareFunction(name, std::move(types))) {
      const char *diagnostic = diag::func_redefinition;
      char buffer[50];
      sprintf(buffer, diagnostic, name.c_str());
      return emitError(node->id().get(), buffer);
    }
    return success();
  }

  auto sema(const StructDecl *node) -> CherryResult {
    llvm::SmallVector<std::string, 2> types;
    llvm::SmallSet<std::string, 4> variables;
    for (auto &varDecl : *node) {
      auto type = varDecl->type().get();
      auto var = varDecl->variable().get();
      if (symbols.checkType(type->name()))
        return emitError(type, diag::type_undefined);
      if (variables.count(var->name()) > 0)
        return emitError(var, diag::var_redefinition);
      variables.insert(var->name());
      types.push_back(type->name());
    }
    auto id = node->id().get();
    if (symbols.declareType(id, std::move(types)))
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
    case Expr::Expr_Struct:
      return sema(cast<StructExpr>(node), type);
    default:
      llvm_unreachable("Unexpected expression");
    }
  }

  auto sema(const VariableExpr *node, std::string& type) -> CherryResult {
    if (symbols.getVariableType(node, type))
      return emitError(node, diag::var_undefined);
    return success();
  }

  auto sema(const CallExpr *node, std::string& type) -> CherryResult {
    auto name = node->name();
    llvm::ArrayRef<std::string> parametersTypes;
    if (symbols.getFunction(name, parametersTypes)) {
      const char * diagnostic = diag::func_undefined;
      char buffer [50];
      sprintf(buffer, diagnostic, name.c_str());
      return emitError(node, buffer);
    }

    auto &expressions = node->expressions();
    if (expressions.size() != parametersTypes.size()) {
      const char * diagnostic = diag::func_param;
      char buffer [50];
      sprintf(buffer, diagnostic, name.c_str(), parametersTypes.size());
      return emitError(node, buffer);
    }

    for (const auto &expr_type : llvm::zip(expressions, parametersTypes)) {
      auto &expr = std::get<0>(expr_type);
      auto type = std::get<1>(expr_type);
      std::string exprType;
      if (sema(expr.get(), exprType))
        return failure();
      if (exprType != type)
        return emitError(expr.get(), diag::func_param_type_mismatch);
    }

    type = symbols.UInt64Type;
    return success();
  }

  auto sema(const DecimalExpr *node, std::string& type) -> CherryResult {
    type = symbols.UInt64Type;
    return success();
  }

  auto sema(const StructExpr *node, std::string& type) -> CherryResult {
    auto typeName = node->type();
    llvm::ArrayRef<std::string> fieldsTypes;
    if (symbols.getType(typeName, fieldsTypes))
      return emitError(node, diag::type_undefined);

    if (node->expressions().size() != fieldsTypes.size())
      return emitError(node, diag::wrong_num_arg);

    for (const auto &expr_type : llvm::zip(*node, fieldsTypes)) {
      auto &expr = std::get<0>(expr_type);
      auto &fieldType = std::get<1>(expr_type);
      std::string exprType;
      if (sema(expr.get(), exprType))
        return failure();
      if (exprType != fieldType)
        return emitError(expr.get(), diag::func_param_type_mismatch);
    }

    type = node->type();
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
