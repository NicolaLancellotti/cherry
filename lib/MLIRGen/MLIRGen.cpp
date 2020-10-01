//===--- MLIRGen.cpp - MLIR Generator -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "StructType.h"
#include "cherry/MLIRGen/CherryOps.h"
#include "cherry/MLIRGen/MLIRGen.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/CherryResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopedHashTable.h"
#include <map>

namespace {
using namespace mlir::cherry;
using namespace cherry;
using llvm::cast;
using mlir::failure;
using mlir::success;

class MLIRGenImpl {
public:
  MLIRGenImpl(const llvm::SourceMgr &sourceManager, mlir::MLIRContext &context)
      : _sourceManager{sourceManager},
        _builder(&context),
        _fileNameIdentifier{_builder.getIdentifier(
            _sourceManager.getMemoryBuffer(_sourceManager.getMainFileID())
                ->getBufferIdentifier())} {}

  auto gen(const Module &node) -> CherryResult;

  mlir::ModuleOp module;
private:
  const llvm::SourceMgr &_sourceManager;
  mlir::OpBuilder _builder;
  std::map<llvm::StringRef, mlir::Value> _variableSymbols;
  std::map<llvm::StringRef, mlir::Type> _typeSymbols;
  mlir::Identifier _fileNameIdentifier;

  // Declarations
  auto gen(const Decl *node, mlir::Operation *&op) -> CherryResult;
  auto gen(const Prototype *node, mlir::FuncOp &func) -> CherryResult;
  auto gen(const FunctionDecl *node, mlir::FuncOp &func) -> CherryResult;
  auto gen(const StructDecl *node) -> CherryResult;

  // Expressions
  auto gen(const Expr *node, mlir::Value &value) -> CherryResult;
  auto genPrint(const CallExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const CallExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const VariableExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const DecimalExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const BinaryExpr *node, mlir::Value &value) -> CherryResult;
  auto genAssign(const BinaryExpr *node, mlir::Value &value) -> CherryResult;

  // Utility
  auto loc(const Node *node) -> mlir::Location {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return _builder.getFileLineColLoc(_fileNameIdentifier, line, col);
  }

  auto getType(llvm::StringRef name) -> mlir::Type {
    if (name == "UInt64") {
      return _builder.getI64Type();
    } else {
      return _typeSymbols[name];
    }
  }
};

} // end namespace

auto MLIRGenImpl::gen(const Module &node) -> CherryResult {
  module = mlir::ModuleOp::create(_builder.getUnknownLoc());

  for (auto &decl : node) {
    mlir::Operation *op;
    if (gen(decl.get(), op))
      return failure();
    if (op)
      module.push_back(op);
  }

  if (failed(mlir::verify(module))) {
    module.emitError("module verification error");
    return failure();
  }

  return success();
}

auto MLIRGenImpl::gen(const Decl *node, mlir::Operation *&op) -> CherryResult {
  switch (node->getKind()) {
  case Decl::Decl_Function: {
    mlir::FuncOp func;
    if (gen(cast<FunctionDecl>(node), func))
      return failure();
    op = func;
    return success();
  }
  case Decl::Decl_Struct: {
    if (gen(cast<StructDecl>(node)))
      return failure();
    op = nullptr;
    return success();
  }
  default:
    llvm_unreachable("Unexpected declaration");
  }
}

auto MLIRGenImpl::gen(const Prototype *node, mlir::FuncOp &func) -> CherryResult {
  mlir::Type i64Type = _builder.getI64Type();

  llvm::SmallVector<mlir::Type, 3> arg_types;
  arg_types.reserve(node->parameters().size());
  for (auto &param : node->parameters())
    arg_types.push_back(getType(param->type()->name()));

  llvm::SmallVector<mlir::Type, 1> result_types(1, i64Type);

  auto funcType = _builder.getFunctionType(arg_types, result_types);
  func = mlir::FuncOp::create(loc(node), node->id()->name(), funcType);

  auto &entryBlock = *func.addEntryBlock();
  for (const auto &var_value : llvm::zip(node->parameters(),
                                         entryBlock.getArguments())) {
    auto name = std::get<0>(var_value)->variable()->name();
    auto value = std::get<1>(var_value);
    _variableSymbols[name] = value;
  }

  _builder.setInsertionPointToStart(&entryBlock);
  return success();
}

auto MLIRGenImpl::gen(const FunctionDecl *node,
                      mlir::FuncOp &func) -> CherryResult {
  _variableSymbols = {};
  if (gen(node->proto().get(), func))
    return failure();

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

auto MLIRGenImpl::gen(const StructDecl *node) -> CherryResult {
  auto &variables = node->variables();
  if (variables.size() == 0) {
    _typeSymbols[node->id()->name()] = _builder.getNoneType();
    return success();
  }

  llvm::SmallVector<mlir::Type, 2> elementTypes;
  elementTypes.reserve(variables.size());
  for (auto &variable : variables) {
    mlir::Type type = getType(variable->type()->name());
    elementTypes.push_back(type);
  }

  _typeSymbols[node->id()->name()] = StructType::get(elementTypes);
  return success();
}

auto MLIRGenImpl::gen(const Expr *node, mlir::Value &value) -> CherryResult {
  switch (node->getKind()) {
  case Expr::Expr_Decimal:
    return gen(cast<DecimalExpr>(node), value);
  case Expr::Expr_Call:
    return gen(cast<CallExpr>(node), value);
  case Expr::Expr_Variable:
    return gen(cast<VariableExpr>(node), value);
  case Expr::Expr_Binary:
    return gen(cast<BinaryExpr>(node), value);
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto MLIRGenImpl::gen(const CallExpr *node, mlir::Value &value) -> CherryResult {
  if (node->name() == "print")
    return genPrint(node, value);

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

auto MLIRGenImpl::genPrint(const CallExpr *node, mlir::Value &value) -> CherryResult {
  auto &expressions = node->expressions();
  mlir::Value operand;
  if (gen(expressions.front().get(), operand))
    return failure();

  value = _builder.create<PrintOp>(loc(node), operand);
  return success();
}

auto MLIRGenImpl::gen(const VariableExpr *node, mlir::Value &value) -> CherryResult {
  value = _variableSymbols[node->name()];
  return success();
}

auto MLIRGenImpl::gen(const DecimalExpr *node, mlir::Value &value) -> CherryResult {
  value = _builder.create<ConstantOp>(loc(node), node->value());
  return success();
}

auto MLIRGenImpl::gen(const BinaryExpr *node, mlir::Value &value) -> CherryResult {
  auto op = node->op();
  if (op == "=")
    return genAssign(node, value);
  else
    llvm_unreachable("Unexpected BinaryExpr operator");
}

auto MLIRGenImpl::genAssign(const BinaryExpr *node, mlir::Value &value) -> CherryResult {
  mlir::Value rhsValue;
  if (gen(node->rhs().get(), rhsValue))
    return failure();

  // TODO: handle struct access
  auto lhs = static_cast<VariableExpr *>(node->lhs().get());
  auto name = lhs->name();

  _variableSymbols[name] =  rhsValue;
  value = rhsValue;
  return success();
}

namespace cherry {

auto mlirGen(const llvm::SourceMgr &sourceManager,
             mlir::MLIRContext &context,
             const Module &moduleAST,
             mlir::OwningModuleRef &module) -> CherryResult {
  auto generator = MLIRGenImpl(sourceManager, context);
  auto result = generator.gen(moduleAST);
  module = generator.module;
  return result;
}

} // end namespace cherry