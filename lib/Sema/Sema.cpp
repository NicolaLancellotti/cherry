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

  auto sema(Module &node) -> CherryResult {
    for (auto &decl : node)
      if (sema(decl.get()))
        return failure();

    llvm::ArrayRef<llvm::StringRef> types;
    llvm::StringRef returnType;
    if (_symbols.getFunction("main", types, returnType) ||
        types.size() != 0 || returnType != builtins::UInt64Type)
      return emitError(llvm::SMLoc{}, diag::main_undefined);
    return success();
  }

private:
  const llvm::SourceMgr &_sourceManager;
  Symbols _symbols;

  // Semantic Analysis

  // Declarations
  auto sema(Decl *node) -> CherryResult;
  auto sema(Prototype *node) -> CherryResult;
  auto sema(FunctionDecl *node) -> CherryResult;
  auto sema(StructDecl *node) -> CherryResult;

  // Expressions
  auto sema(Expr *node) -> CherryResult;
  auto sema(CallExpr *node) -> CherryResult;
  auto semaStructConstructor(CallExpr *node) -> CherryResult;
  auto sema(VariableDeclExpr *node) -> CherryResult;
  auto sema(VariableExpr *node) -> CherryResult;
  auto sema(DecimalLiteralExpr *node) -> CherryResult;
  auto sema(BoolLiteralExpr *node) -> CherryResult;
  auto sema(BinaryExpr *node) -> CherryResult;
  auto semaAssign(BinaryExpr *node) -> CherryResult;
  auto semaStructAccess(BinaryExpr *node) -> CherryResult;

  // Errors
  auto emitError(Node *node, const llvm::Twine &msg) -> CherryResult {
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

auto SemaImpl::sema(Decl *node) -> CherryResult {
  switch (node->getKind()) {
  case Decl::Decl_Function:
    return sema(cast<FunctionDecl>(node));
  case Decl::Decl_Struct:
    return sema(cast<StructDecl>(node));
  default:
    llvm_unreachable("Unexpected declaration");
  }
}

auto SemaImpl::sema(Prototype *node) -> CherryResult {
  llvm::SmallVector<llvm::StringRef, 2> types;
  for (auto &par : node->parameters()) {
    auto type = par->varType().get();
    auto typeName = type->name();
    if (_symbols.checkType(typeName))
      return emitError(type, diag::type_undefined);
    if (_symbols.declareVariable(par->variable().get(), typeName))
      return emitError(par->variable().get(), diag::var_redefinition);
    types.push_back(typeName);
  }

  auto returnType = node->type()->name();
  if (_symbols.checkType(returnType))
    return emitError(node->type().get(), diag::type_undefined);

  auto name = node->id()->name();
  if (_symbols.declareFunction(name, std::move(types), returnType)) {
    const char *diagnostic = diag::func_redefinition;
    char buffer[50];
    sprintf(buffer, diagnostic, name.str().c_str());
    return emitError(node->id().get(), buffer);
  }
  return success();
}

auto SemaImpl::sema(FunctionDecl *node) -> CherryResult {
  _symbols.resetVariables();
  if (sema(node->proto().get()))
    return failure();

  for (auto &expr : *node)
    if (sema(expr.get()))
      return failure();

  auto returnType = node->body().back()->type();
  if (returnType != node->proto()->type()->name())
    return emitError(node->body().back().get(), diag::wrong_return_type);

  return success();
}

auto SemaImpl::sema(StructDecl *node) -> CherryResult {
  llvm::SmallVector<llvm::StringRef, 2> types;
  llvm::SmallSet<llvm::StringRef, 4> variables;
  for (auto &varDecl : *node) {
    auto type = varDecl->varType().get();
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

auto SemaImpl::sema(Expr *node) -> CherryResult {
  switch (node->getKind()) {
  case Expr::Expr_DecimalLiteral:
    return sema(cast<DecimalLiteralExpr>(node));
  case Expr::Expr_BoolLiteral:
    return sema(cast<BoolLiteralExpr>(node));
  case Expr::Expr_Call:
    return sema(cast<CallExpr>(node));
  case Expr::Expr_VariableDecl:
    return sema(cast<VariableDeclExpr>(node));
  case Expr::Expr_Variable:
    return sema(cast<VariableExpr>(node));
  case Expr::Expr_Binary:
    return sema(cast<BinaryExpr>(node));
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto SemaImpl::sema(CallExpr *node) -> CherryResult {
  auto name = node->name();

  if (mlir::succeeded(_symbols.checkType(name)))
    return semaStructConstructor(node);

  llvm::ArrayRef<llvm::StringRef> parametersTypes;
  llvm::StringRef returnType;
  if (_symbols.getFunction(name, parametersTypes, returnType)) {
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
    if (sema(expr.get()))
      return failure();
    if (expr->type() != type)
      return emitError(expr.get(), diag::type_mismatch);
  }

  node->setType(returnType);
  return success();
}

auto SemaImpl::semaStructConstructor(CallExpr *node) -> CherryResult {
  auto type = node->name();
  const VectorUniquePtr<VariableDeclExpr> *fieldsTypes;
  if (_symbols.getType(type, fieldsTypes))
    return emitError(node, diag::type_undefined);

  if (node->expressions().size() != fieldsTypes->size())
    return emitError(node, diag::wrong_num_arg);

  for (const auto &expr_type : llvm::zip(*node, *fieldsTypes)) {
    auto &expr = std::get<0>(expr_type);
    auto fieldType = std::get<1>(expr_type)->varType()->name();
    if (sema(expr.get()))
      return failure();
    if (expr->type() != fieldType)
      return emitError(expr.get(), diag::type_mismatch);
  }

  node->setType(type);
  return success();
}

auto SemaImpl::sema(VariableDeclExpr *node) -> CherryResult {
  auto var = node->variable().get();
  auto varType = node->varType().get();
  auto varTypeName = varType->name();
  if (_symbols.checkType(varTypeName))
    return emitError(varType, diag::type_undefined);
  if (_symbols.declareVariable(var, varTypeName))
    return emitError(var, diag::var_redefinition);

  auto initExpr = node->init().get();
  if (sema(initExpr))
    return failure();
  if (varTypeName != initExpr->type())
    return emitError(initExpr, diag::type_mismatch);

  node->setType(builtins::UInt64Type);
  return success();
}

auto SemaImpl::sema(VariableExpr *node) -> CherryResult {
  llvm::StringRef type;
  if (_symbols.getVariableType(node, type))
    return emitError(node, diag::var_undefined);
  node->setType(type);
  return success();
}

auto SemaImpl::sema(DecimalLiteralExpr *node) -> CherryResult {
  node->setType(builtins::UInt64Type);
  return success();
}

auto SemaImpl::sema(BoolLiteralExpr *node) -> CherryResult {
  node->setType(builtins::BoolType);
  return success();
}

auto SemaImpl::sema(BinaryExpr *node) -> CherryResult {
  auto op = node->op();
  if (op == "=")
    return semaAssign(node);
  else if (op == ".")
    return semaStructAccess(node);
  else
    llvm_unreachable("Unexpected BinaryExpr operator");
}

auto SemaImpl::semaAssign(BinaryExpr *node) -> CherryResult {
  if (sema(node->lhs().get()) || sema(node->rhs().get()))
    return failure();

  if (!node->lhs()->isLvalue())
    return emitError(node->lhs().get(), diag::expected_lvalue);

  auto lhsType = node->lhs()->type();
  auto rhsType = node->rhs()->type();
  if (lhsType != rhsType)
    return emitError(node->rhs().get(), diag::type_mismatch);

  node->setType(lhsType);
  return success();
}

auto SemaImpl::semaStructAccess(BinaryExpr *node) -> CherryResult {
  if (sema(node->lhs().get()))
    return failure();
  VariableExpr *var = llvm::dyn_cast<VariableExpr>(node->rhs().get());
  if (!var)
    return emitError(node->rhs().get(), diag::expected_field);

  auto fieldName = var->name();
  const VectorUniquePtr<VariableDeclExpr> *fieldsTypes;
  auto lhsType = node->lhs()->type();
  _symbols.getType(lhsType, fieldsTypes);

  for (auto &f : *fieldsTypes) {
    if (f->variable()->name() == fieldName) {
      node->setType(f->varType()->name());
      return success();
    }
  }

  return emitError(node->rhs().get(), diag::field_undefined);
}

namespace cherry {

auto sema(const llvm::SourceMgr &sourceManager,
          Module &moduleAST) -> CherryResult {
  return SemaImpl(sourceManager).sema(moduleAST);
}

} // end namespace cherry
