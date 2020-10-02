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
    _symbols.addBuiltins();
  }

  auto sema(const Module &node) -> CherryResult {
    for (auto &decl : node)
      if (sema(decl.get()))
        return failure();

    llvm::ArrayRef<llvm::StringRef> types;
    if (_symbols.getFunction("main", types) || types.size() != 0)
      return emitError(llvm::SMLoc{}, diag::main_undefined);
    return success();
  }

private:
  const llvm::SourceMgr &_sourceManager;
  Symbols _symbols;

  // Semantic Analysis

  // Declarations
  auto sema(const Decl *node) -> CherryResult;
  auto sema(const Prototype *node) -> CherryResult;
  auto sema(const FunctionDecl *node) -> CherryResult;
  auto sema(const StructDecl *node) -> CherryResult;

  // Expressions
  auto sema(const Expr *node, llvm::StringRef &type) -> CherryResult;
  auto sema(const CallExpr *node, llvm::StringRef &type) -> CherryResult;
  auto sema(const VariableDeclExpr *node, llvm::StringRef &type) -> CherryResult;
  auto sema(const VariableExpr *node, llvm::StringRef &type) -> CherryResult;
  auto sema(const DecimalLiteralExpr *node, llvm::StringRef &type) -> CherryResult;
  auto sema(const BoolLiteralExpr *node, llvm::StringRef &type) -> CherryResult;
  auto sema(const StructExpr *node, llvm::StringRef &type) -> CherryResult;
  auto sema(const BinaryExpr *node, llvm::StringRef &type) -> CherryResult;
  auto semaAssign(const BinaryExpr *node, llvm::StringRef &type) -> CherryResult;
  auto semaStructAccess(const BinaryExpr *node,
                        llvm::StringRef &type) -> CherryResult;

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

};

} // end namespace

auto SemaImpl::sema(const Decl *node) -> CherryResult {
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

auto SemaImpl::sema(const Prototype *node) -> CherryResult {
  llvm::SmallVector<llvm::StringRef, 2> types;
  for (auto &par : node->parameters()) {
    auto type = par->type().get();
    auto typeName = type->name();
    if (_symbols.checkType(typeName))
      return emitError(type, diag::type_undefined);
    if (_symbols.declareVariable(par->variable().get(), typeName))
      return emitError(par->variable().get(), diag::var_redefinition);
    types.push_back(typeName);
  }

  auto name = node->id()->name();
  if (_symbols.declareFunction(name, std::move(types))) {
    const char *diagnostic = diag::func_redefinition;
    char buffer[50];
    sprintf(buffer, diagnostic, name.str().c_str());
    return emitError(node->id().get(), buffer);
  }
  return success();
}

auto SemaImpl::sema(const FunctionDecl *node) -> CherryResult {
  _symbols.resetVariables();
  if (sema(node->proto().get()))
    return failure();

  llvm::StringRef type;
  for (auto &expr : *node)
    if (sema(expr.get(), type))
      return failure();

  if (type != builtins::UInt64Type)
    return emitError(node->body().back().get(), diag::wrong_return_type);

  return success();
}

auto SemaImpl::sema(const StructDecl *node) -> CherryResult {
  llvm::SmallVector<llvm::StringRef, 2> types;
  llvm::SmallSet<llvm::StringRef, 4> variables;
  for (auto &varDecl : *node) {
    auto type = varDecl->type().get();
    auto var = varDecl->variable().get();
    if (_symbols.checkType(type->name()))
      return emitError(type, diag::type_undefined);
    if (variables.count(var->name()) > 0)
      return emitError(var, diag::var_redefinition);
    variables.insert(var->name());
    types.push_back(type->name());
  }
  auto id = node->id().get();
  if (_symbols.declareType(node))
    return emitError(id, diag::type_redefinition);
  return success();
}

auto SemaImpl::sema(const Expr *node, llvm::StringRef &type) -> CherryResult {
  switch (node->getKind()) {
  case Expr::Expr_DecimalLiteral:
    return sema(cast<DecimalLiteralExpr>(node), type);
  case Expr::Expr_BoolLiteral:
    return sema(cast<BoolLiteralExpr>(node), type);
  case Expr::Expr_Call:
    return sema(cast<CallExpr>(node), type);
  case Expr::Expr_VariableDecl:
    return sema(cast<VariableDeclExpr>(node), type);
  case Expr::Expr_Variable:
    return sema(cast<VariableExpr>(node), type);
  case Expr::Expr_Struct:
    return sema(cast<StructExpr>(node), type);
  case Expr::Expr_Binary:
    return sema(cast<BinaryExpr>(node), type);
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto SemaImpl::sema(const CallExpr *node, llvm::StringRef &type) -> CherryResult {
  auto name = node->name();
  llvm::ArrayRef<llvm::StringRef> parametersTypes;
  if (_symbols.getFunction(name, parametersTypes)) {
    const char *diagnostic = diag::func_undefined;
    char buffer [50];
    sprintf(buffer, diagnostic, name.str().c_str());
    return emitError(node, buffer);
  }

  auto &expressions = node->expressions();
  if (expressions.size() != parametersTypes.size()) {
    const char *diagnostic = diag::func_param;
    char buffer [50];
    sprintf(buffer, diagnostic, name.str().c_str(), parametersTypes.size());
    return emitError(node, buffer);
  }

  for (const auto &expr_type : llvm::zip(expressions, parametersTypes)) {
    auto &expr = std::get<0>(expr_type);
    auto type = std::get<1>(expr_type);
    llvm::StringRef exprType;
    if (sema(expr.get(), exprType))
      return failure();
    if (exprType != type)
      return emitError(expr.get(), diag::type_mismatch);
  }

  type = builtins::UInt64Type;
  return success();
}

auto SemaImpl::sema(const VariableDeclExpr *node, llvm::StringRef &type) -> CherryResult {
  auto var = node->variable().get();
  auto varType = node->type().get();
  auto varTypeName = varType->name();
  if (_symbols.checkType(varTypeName))
    return emitError(varType, diag::type_undefined);
  if (_symbols.declareVariable(var, varTypeName))
    return emitError(var, diag::var_redefinition);

  auto initValue = node->init().get();
  llvm::StringRef initValueType;
  if (sema(initValue, initValueType))
    return failure();

  if (varTypeName != initValueType)
    return emitError(initValue, diag::type_mismatch);

  return success();
}

auto SemaImpl::sema(const VariableExpr *node, llvm::StringRef &type) -> CherryResult {
  if (_symbols.getVariableType(node, type))
    return emitError(node, diag::var_undefined);
  return success();
}

auto SemaImpl::sema(const DecimalLiteralExpr *node, llvm::StringRef &type) -> CherryResult {
  type = builtins::UInt64Type;
  return success();
}

auto SemaImpl::sema(const BoolLiteralExpr *node, llvm::StringRef &type) -> CherryResult {
  type = builtins::BoolType;
  return success();
}

auto SemaImpl::sema(const StructExpr *node, llvm::StringRef &type) -> CherryResult {
  auto typeName = node->type();
  const VectorUniquePtr<VariableDeclExpr> *fieldsTypes;
  if (_symbols.getType(typeName, fieldsTypes))
    return emitError(node, diag::type_undefined);

  if (node->expressions().size() != fieldsTypes->size())
    return emitError(node, diag::wrong_num_arg);

  for (const auto &expr_type : llvm::zip(*node, *fieldsTypes)) {
    auto &expr = std::get<0>(expr_type);
    auto fieldType = std::get<1>(expr_type)->type()->name();
    llvm::StringRef exprType;
    if (sema(expr.get(), exprType))
      return failure();
    if (exprType != fieldType)
      return emitError(expr.get(), diag::type_mismatch);
  }

  type = node->type();
  return success();
}

auto SemaImpl::sema(const BinaryExpr *node,
                    llvm::StringRef &type) -> CherryResult {
  auto op = node->op();
  if (op == "=")
    return semaAssign(node, type);
  else if (op == ".")
    return semaStructAccess(node, type);
  else
    llvm_unreachable("Unexpected BinaryExpr operator");
}

auto SemaImpl::semaAssign(const BinaryExpr *node,
                          llvm::StringRef &type) -> CherryResult {
  llvm::StringRef lhsType;
  llvm::StringRef rhsType;
  if (sema(node->lhs().get(), lhsType) || sema(node->rhs().get(), rhsType))
    return failure();
  if (!node->lhs()->isLvalue())
    return emitError(node->lhs().get(), diag::expected_lvalue);
  if (lhsType != rhsType)
    return emitError(node->rhs().get(), diag::type_mismatch);
  type = lhsType;
  return success();
}

auto SemaImpl::semaStructAccess(const BinaryExpr *node,
                                llvm::StringRef &type) -> CherryResult {
  llvm::StringRef lhsType;
  if (sema(node->lhs().get(), lhsType))
    return failure();
  VariableExpr *var = llvm::dyn_cast<VariableExpr>(node->rhs().get());
  if (!var)
    return emitError(node->rhs().get(), diag::expected_field);

  auto fieldName = var->name();
  const VectorUniquePtr<VariableDeclExpr> *fieldsTypes;
  _symbols.getType(lhsType, fieldsTypes);

  for (auto &f : *fieldsTypes) {
    if (f->variable()->name() == fieldName) {
      type = f->type()->name();
      return success();
    }
  }

  return emitError(node->rhs().get(), diag::field_undefined);
}

namespace cherry {

auto sema(const llvm::SourceMgr &sourceManager,
          const Module &moduleAST) -> CherryResult {
  return SemaImpl(sourceManager).sema(moduleAST);
}

} // end namespace cherry
