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
  std::map<std::string, int> _symbols;

  auto addBuiltin() -> void {
    _symbols.insert(make_pair("print", 1));
  }

  auto emitError(const Node *node, const llvm::Twine &msg) -> CherryResult {
    _sourceManager.PrintMessage(node->location(),
                                llvm::SourceMgr::DiagKind::DK_Error,
                                msg);
    return failure();
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
    if (sema(node->proto().get()))
      return failure();

    for (auto &expr : *node) {
      if (sema(expr.get()))
        return failure();
    }
    return success();
  }

  auto sema(const Prototype *node) -> CherryResult {
    auto name = node->name();
    auto result = _symbols.insert(make_pair(name, 0));
    if (result.second)
      return success();

    const char * diagnostic = diag::func_redefinition;
    char buffer [50];
    sprintf(buffer, diagnostic, name.c_str());
    return emitError(node, buffer);
  }

  auto sema(const Expr *node) -> CherryResult {
    switch (node->getKind()) {
    case Expr::Expr_Decimal:
      return sema(cast<DecimalExpr>(node));
    case Expr::Expr_Call:
      return sema(cast<CallExpr>(node));
    default:
      return failure();
    }
  }

  auto sema(const CallExpr *node) -> CherryResult {
    auto name = node->name();
    auto symbol = _symbols.find(name);
    if (symbol == _symbols.end()) {
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
