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
    addBuiltin();
  }

  auto sema(const Module &node) -> CherryResult {
    for (auto &decl : node) {
      if (sema(decl.get()))
        return failure();
    }
    return success();
  }

private:
  const llvm::SourceMgr &_sourceManager;
  std::map<std::string, int> _functionSymbols;
  std::set<std::string> _typeSymbols;
  std::set<llvm::StringRef> _variableSymbols;

  auto addBuiltin() -> void {
    _functionSymbols.insert(make_pair("print", 1));
    _typeSymbols.insert("UInt64");
  }

  auto declareVariable(llvm::StringRef name) -> CherryResult {
    if (_variableSymbols.find(name) != _variableSymbols.end())
      return failure();
    _variableSymbols.insert(name);
    return success();
  }

  auto emitError(const Node *node, const llvm::Twine &msg) -> CherryResult {
    _sourceManager.PrintMessage(node->location(),
                                llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
  }

  auto checkTypeExist(const Variable *node) -> CherryResult {
    if (_typeSymbols.find(node->name()) == _typeSymbols.end())
      return failure();
    return success();
  }

  auto sema(const Decl *node) -> CherryResult {
    switch (node->getKind()) {
    case Decl::Decl_Function: {
      return sema(cast<FunctionDecl>(node));
    }
    default:
      return failure();
    }
  }

  auto sema(const FunctionDecl *node) -> CherryResult {
    _variableSymbols = {};
    if (sema(node->proto().get()))
      return failure();

    for (auto &expr : *node) {
      if (sema(expr.get()))
        return failure();
    }
    return success();
  }

  auto sema(const Prototype *node) -> CherryResult {
    for (auto &par : node->parameters()) {
      if (declareVariable(par.first->name()))
        return emitError(par.first.get(), diag::var_redefinition);
      if (checkTypeExist(par.second.get()))
        return emitError(par.second.get(), diag::type_undefined);
    }

    auto name = node->id()->name();
    auto result = _functionSymbols.insert(make_pair(name,
                                                    node->parameters().size()));
    if (!result.second) {
      const char *diagnostic = diag::func_redefinition;
      char buffer[50];
      sprintf(buffer, diagnostic, name.c_str());
      return emitError(node->id().get(), buffer);
    }
    return success();
  }

  auto sema(const Expr *node) -> CherryResult {
    switch (node->getKind()) {
    case Expr::Expr_Decimal:
      return sema(cast<DecimalExpr>(node));
    case Expr::Expr_Call:
      return sema(cast<CallExpr>(node));
    case Expr::Expr_Variable:
      return sema(cast<Variable>(node));
    default:
      return failure();
    }
  }

  auto sema(const Variable *node) -> CherryResult {
    if (_variableSymbols.find(node->name()) == _variableSymbols.end())
      return emitError(node, diag::var_undefined);
    return success();
  }

  auto sema(const CallExpr *node) -> CherryResult {
    auto name = node->name();
    auto symbol = _functionSymbols.find(name);
    if (symbol == _functionSymbols.end()) {
      const char * diagnostic = diag::func_undefined;
      char buffer [50];
      sprintf(buffer, diagnostic, name.c_str());
      return emitError(node, buffer);
    }

    auto formalParameters = symbol->second;
    auto actualParameters = node->expressions().size();
    if (actualParameters != formalParameters) {
      const char * diagnostic = diag::func_param;
      char buffer [50];
      sprintf(buffer, diagnostic, name.c_str(), formalParameters);
      return emitError(node, buffer);
    }

    for (auto &expr : *node) {
      if (sema(expr.get()))
        return failure();
    }
    return success();
  }

  auto sema(const DecimalExpr *node) -> CherryResult {
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
