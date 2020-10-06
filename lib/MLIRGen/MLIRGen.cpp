//===--- MLIRGen.cpp - MLIR Generator -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/MLIRGen.h"
#include "StructType.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/Builtins.h"
#include "cherry/Basic/CherryResult.h"
#include "cherry/MLIRGen/CherryOps.h"
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
  std::map<llvm::StringRef, mlir::Type> _functionSymbols;
  mlir::Identifier _fileNameIdentifier;

  // Declarations
  auto gen(const Decl *node, mlir::Operation *&op) -> CherryResult;
  auto gen(const Prototype *node, mlir::FuncOp &func) -> CherryResult;
  auto gen(const FunctionDecl *node, mlir::FuncOp &func) -> CherryResult;
  auto gen(const StructDecl *node) -> CherryResult;

  // Expressions
  auto gen(const Expr *node, mlir::Value &value) -> CherryResult;
  auto gen(const VectorUniquePtr<Expr> &node, mlir::Value &value) -> CherryResult;
  auto gen(const IfExpr *node, mlir::Value &value) -> CherryResult;
  auto genPrint(const CallExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const CallExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const VariableDeclExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const VariableExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const DecimalLiteralExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const BoolLiteralExpr *node, mlir::Value &value) -> CherryResult;
  auto gen(const BinaryExpr *node, mlir::Value &value) -> CherryResult;
  auto genAssign(const BinaryExpr *node, mlir::Value &value) -> CherryResult;

  // Utility
  auto loc(const Node *node) -> mlir::Location {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return _builder.getFileLineColLoc(_fileNameIdentifier, line, col);
  }

  auto getType(llvm::StringRef name) -> mlir::Type {
    if (name == builtins::UInt64Type) {
      return _builder.getI64Type();
    } else if (name == builtins::BoolType) {
      return _builder.getI1Type();
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
  llvm::SmallVector<mlir::Type, 3> arg_types;
  arg_types.reserve(node->parameters().size());
  for (auto &param : node->parameters())
    arg_types.push_back(getType(param->varType()->name()));

  llvm::SmallVector<mlir::Type, 1> result_types(1, getType(node->type()->name()));

  auto funcName = node->id()->name();
  auto funcType = _builder.getFunctionType(arg_types, result_types);
  func = mlir::FuncOp::create(loc(node), funcName, funcType);

  auto &entryBlock = *func.addEntryBlock();
  for (const auto &var_value : llvm::zip(node->parameters(),
                                         entryBlock.getArguments())) {
    auto name = std::get<0>(var_value)->variable()->name();
    auto value = std::get<1>(var_value);
    _variableSymbols[name] = value;
  }

  _builder.setInsertionPointToStart(&entryBlock);
  _functionSymbols[funcName] = getType(node->type()->name());
  return success();
}

auto MLIRGenImpl::gen(const FunctionDecl *node,
                      mlir::FuncOp &func) -> CherryResult {
  _variableSymbols = {};
  if (gen(node->proto().get(), func))
    return failure();

  mlir::Value value;
  if (gen(node->body(), value)) {
    func.erase();
    return failure();
  }

  auto location = loc(node->body().back().get());
  _builder.create<ReturnOp>(location, value);
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
    mlir::Type type = getType(variable->varType()->name());
    elementTypes.push_back(type);
  }

  _typeSymbols[node->id()->name()] = StructType::get(elementTypes);
  return success();
}

auto MLIRGenImpl::gen(const Expr *node, mlir::Value &value) -> CherryResult {
  switch (node->getKind()) {
  case Expr::Expr_DecimalLiteral:
    return gen(cast<DecimalLiteralExpr>(node), value);
  case Expr::Expr_BoolLiteral:
    return gen(cast<BoolLiteralExpr>(node), value);
  case Expr::Expr_Call:
    return gen(cast<CallExpr>(node), value);
  case Expr::Expr_VariableDecl:
    return gen(cast<VariableDeclExpr>(node), value);
  case Expr::Expr_Variable:
    return gen(cast<VariableExpr>(node), value);
  case Expr::Expr_Binary:
    return gen(cast<BinaryExpr>(node), value);
  case Expr::Expr_If:
    return gen(cast<IfExpr>(node), value);
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto MLIRGenImpl::gen(const VectorUniquePtr<Expr> &node, mlir::Value &value) -> CherryResult {
  for (auto &expr : node)
    if (gen(expr.get(), value))
      return failure();
  return success();
}

auto MLIRGenImpl::gen(const IfExpr *node, mlir::Value &value) -> CherryResult {
  mlir::Value cond;
  if (gen(node->conditionExpr().get(), cond))
    return failure();

  bool error = false;

  auto &thenExpr = node->thenExpr();
  auto thenExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    mlir::Value value;
    if (gen(thenExpr, value))
      error = true;
    builder.create<YieldOp>(loc(thenExpr.back().get()), value);
  };

  auto &elseExpr = node->elseExpr();
  auto elseExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    mlir::Value value;
    if (gen(elseExpr, value))
      error = true;
    builder.create<YieldOp>(loc(elseExpr.back().get()), value);
  };
  auto ifOp = _builder.create<IfOp>(loc(node),
                                               getType(node->type()),
                                               cond,
                                               thenExprBuilder,
                                               elseExprBuilder);
  if (error)
    return failure();

  value = ifOp.getResult();
  return success();
}

auto MLIRGenImpl::gen(const CallExpr *node, mlir::Value &value) -> CherryResult {
  auto functionName = node->name();
  if (node->name() == builtins::print)
    return genPrint(node, value);

  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto &expr : *node) {
    mlir::Value value;
    if (gen(expr.get(), value))
      return failure();
    operands.push_back(value);
  }
  if (functionName == builtins::boolToUInt64)
    value = _builder.create<CastOp>(loc(node),  operands.front());
  else
    value = _builder.create<CallOp>(loc(node), node->name(), operands,
                                    _functionSymbols[functionName]);
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

auto MLIRGenImpl::gen(const VariableDeclExpr *node,
                      mlir::Value &value) -> CherryResult {
  auto name = node->variable()->name();
  mlir::Value initValue;
  if (gen(node->init().get(), initValue))
    return failure();
  _variableSymbols[name] = initValue;
  value = nullptr;
  return success();
}

auto MLIRGenImpl::gen(const VariableExpr *node, mlir::Value &value) -> CherryResult {
  value = _variableSymbols[node->name()];
  return success();
}

auto MLIRGenImpl::gen(const DecimalLiteralExpr *node, mlir::Value &value) -> CherryResult {
  value = _builder.create<ConstantOp>(loc(node), node->value());
  return success();
}

auto MLIRGenImpl::gen(const BoolLiteralExpr *node, mlir::Value &value) -> CherryResult {
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