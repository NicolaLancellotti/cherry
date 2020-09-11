//===--- Sema.cpp - Cherry Semantic Analysis ------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/CherryResult.h"

namespace {
using namespace cherry;
using llvm::cast;
using mlir::failure;
using mlir::success;

class SemaImpl {
public:
  SemaImpl(const llvm::SourceMgr &sourceManager)
      : _sourceManager{sourceManager} {}

  auto sema(const Module &node) -> CherryResult {
    for (auto &decl : node) {
      if (sema(decl.get()))
        return failure();
    }
    return success();
  }

private:
  const llvm::SourceMgr &_sourceManager;

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
    return success();
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
