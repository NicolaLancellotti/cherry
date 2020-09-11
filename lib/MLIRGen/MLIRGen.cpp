//===--- MLIRGen.cpp - MLIR Generator -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/MLIRGen.h"
#include "cherry/MLIRGen/CherryOps.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/CherryResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/SMLoc.h"

namespace {
using namespace mlir::cherry;
using namespace cherry;
using llvm::cast;
using mlir::failure;
using mlir::success;

class MLIRGenImpl {
public:
  MLIRGenImpl(const llvm::SourceMgr &sourceManager, mlir::MLIRContext &context)
      : _sourceManager{sourceManager}, _builder(&context) {}

  auto gen(const Module &node) -> CherryResult {
    module = mlir::ModuleOp::create(_builder.getUnknownLoc());

    for (auto &decl : node) {
      mlir::Operation *op;
      if (gen(decl.get(), op))
        return failure();
      module.push_back(op);
    }

    if (failed(mlir::verify(module))) {
      module.emitError("module verification error");
      return failure();
    }

    return success();
  }

  mlir::ModuleOp module;
private:
  const llvm::SourceMgr &_sourceManager;
  mlir::OpBuilder _builder;

  auto loc(const Node *node) -> mlir::Location {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    auto identifier = _builder.getIdentifier("main.cherry");
    return _builder.getFileLineColLoc(identifier, line, col);
  }

  auto gen(const Decl *node, mlir::Operation *&op) -> CherryResult {
    switch (node->getKind()) {
    case Decl::Decl_Function: {
      mlir::FuncOp func;
      if (gen(cast<FunctionDecl>(node), func))
        return failure();
      op = func;
      return success();
    }
    default:
      return failure();
    }
  }

  auto gen(const FunctionDecl *node,
           mlir::FuncOp& func) -> CherryResult {
    if (gen(node->proto().get(), func))
      return failure();

    auto &entryBlock = *func.addEntryBlock();
    _builder.setInsertionPointToStart(&entryBlock);

    for (auto &expr : *node) {
      mlir::Value value;
      if (gen(expr.get(), value)) {
        func.erase();
        return failure();
      }
    }
    auto location = loc(node->proto().get());
    ConstantOp constant0 = _builder.create<ConstantOp>(location, 0);
    _builder.create<ReturnOp>(location, constant0);
    return success();
  }

  auto gen(const Prototype *node, mlir::FuncOp& func) -> CherryResult {
    mlir::Type noneType = mlir::NoneType();
    llvm::SmallVector<mlir::Type, 0> arg_types(0, noneType);

    mlir::Type resultType = _builder.getI64Type();
    llvm::SmallVector<mlir::Type, 1> result_types(1, resultType);

    auto funcType = _builder.getFunctionType(arg_types, result_types);
    func = mlir::FuncOp::create(loc(node), node->name(), funcType);
    return success();
  }

  auto gen(const Expr *node, mlir::Value& value) -> CherryResult {
    switch (node->getKind()) {
    case Expr::Expr_Decimal:
      return gen(cast<DecimalExpr>(node), value);
    case Expr::Expr_Call:
      return gen(cast<CallExpr>(node), value);
    default:
      return failure();
    }
  }

  auto genPrint(const CallExpr *node, mlir::Value& value) -> CherryResult {
    auto &expressions = node->expressions();
    if (expressions.size() != 1) {
      emitError(loc(node)) << "print call takes one argument ";
      return failure();
    }

    mlir::Value operand;
    if (gen(expressions.front().get(), operand))
      return failure();

    value = _builder.create<PrintOp>(loc(node), operand);
    return success();
  }

  auto gen(const CallExpr *node, mlir::Value& value) -> CherryResult {
    if (node->name() == "print") {
      return genPrint(node, value);
    }

    llvm::SmallVector<mlir::Value, 4> operands;
    for (auto &expr : *node) {
      mlir::Value value;
      if (gen(expr.get(), value))
        return failure();
      operands.push_back(value);
    }
    value = _builder.create<CallOp>(loc(node), node->name(), operands);
    return success();
  }

  auto gen(const DecimalExpr *node, mlir::Value& value) -> CherryResult {
    value = _builder.create<ConstantOp>(loc(node), node->value());
    return success();
  }

};

} // end namespace

namespace cherry {

auto mlirGen(const llvm::SourceMgr &sourceManager,
             mlir::MLIRContext &context,
             const Module &moduleAST) -> mlir::OwningModuleRef {
  auto generator = MLIRGenImpl(sourceManager, context);
  return generator.gen(moduleAST) ? nullptr : generator.module;
}

} // end namespace cherry