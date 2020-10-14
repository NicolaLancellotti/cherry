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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
  auto gen(const Decl *node, mlir::Operation *&op) -> void;
  auto gen(const Prototype *node, mlir::FuncOp &func) -> void;
  auto gen(const FunctionDecl *node, mlir::FuncOp &func) -> void;
  auto gen(const StructDecl *node) -> void;

  // Expressions
  auto gen(const Expr *node, mlir::Value &value) -> void;
  auto gen(const UnitExpr *node, mlir::Value &value) -> void;
  auto gen(const BlockExpr *node, mlir::Value &value) -> void;
  auto gen(const IfExpr *node, mlir::Value &value) -> void;
  auto gen(const WhileExpr *node, mlir::Value &value) -> void;
  auto genPrint(const CallExpr *node, mlir::Value &value) -> void;
  auto gen(const CallExpr *node, mlir::Value &value) -> void;
  auto gen(const VariableExpr *node, mlir::Value &value) -> void;
  auto gen(const DecimalLiteralExpr *node, mlir::Value &value) -> void;
  auto gen(const BoolLiteralExpr *node, mlir::Value &value) -> void;
  auto gen(const BinaryExpr *node, mlir::Value &value) -> void;
  auto genAssign(const BinaryExpr *node, mlir::Value &value) -> void;

  // Statements
  auto gen(const Stat *node) -> void;
  auto gen(const VariableStat *node) -> void;
  auto gen(const ExprStat *node) -> void;

  // Utility
  auto loc(const Node *node) -> mlir::Location {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return _builder.getFileLineColLoc(_fileNameIdentifier, line, col);
  }

  auto getType(llvm::StringRef name) -> mlir::Type {
    if (name == builtins::UnitType) {
      return nullptr;
    } else if (name == builtins::UInt64Type) {
      return _builder.getI64Type();
    } else if (name == builtins::BoolType) {
      return _builder.getI1Type();
    } else {
      return _typeSymbols[name];
    }
  }

  mlir::Value createEntryBlockAlloca(mlir::Type mlirtType, mlir::Location loc) {
    if (mlirtType == getType(builtins::UnitType))
      return nullptr;
    mlir::MemRefType memRefType = mlir::MemRefType::get({}, mlirtType);
    auto alloca = _builder.create<mlir::AllocaOp>(loc, memRefType);
    auto *parentBlock = alloca.getOperation()->getBlock();
    alloca.getOperation()->moveBefore(&parentBlock->front());
    return alloca;
  }
};

} // end namespace

auto MLIRGenImpl::gen(const Module &node) -> CherryResult {
  module = mlir::ModuleOp::create(_builder.getUnknownLoc());

  for (auto &decl : node) {
    mlir::Operation *op;
    gen(decl.get(), op);
    if (op)
      module.push_back(op);
  }

  if (failed(mlir::verify(module))) {
    module.emitError("module verification error");
    return failure();
  }

  return success();
}

auto MLIRGenImpl::gen(const Decl *node, mlir::Operation *&op) -> void {
  switch (node->getKind()) {
  case Decl::Decl_Function: {
    mlir::FuncOp func;
    gen(cast<FunctionDecl>(node), func);
    op = func;
    return;
  }
  case Decl::Decl_Struct: {
    gen(cast<StructDecl>(node));
    op = nullptr;
    return;
  }
  default:
    llvm_unreachable("Unexpected declaration");
  }
}

auto MLIRGenImpl::gen(const Prototype *node, mlir::FuncOp &func) -> void {
  llvm::SmallVector<mlir::Type, 3> arg_types;
  arg_types.reserve(node->parameters().size());
  for (auto &param : node->parameters())
    arg_types.push_back(getType(param->varType()->name()));

  llvm::SmallVector<mlir::Type, 1> result_types;
  if (auto type = getType(node->type()->name()))
    result_types.push_back(type);

  auto funcName = node->id()->name();
  auto funcType = _builder.getFunctionType(arg_types, result_types);
  func = mlir::FuncOp::create(loc(node), funcName, funcType);

  auto &entryBlock = *func.addEntryBlock();
  _builder.setInsertionPointToStart(&entryBlock);
  for (const auto &var_value : llvm::zip(node->parameters(),
                                         entryBlock.getArguments())) {
    auto &var = std::get<0>(var_value);
    auto varName = var->variable()->name();
    auto typeName = var->varType()->name();
    auto value = std::get<1>(var_value);
    auto alloca = createEntryBlockAlloca(getType(typeName), loc(node));
    _variableSymbols[varName] = alloca;
    _builder.create<mlir::StoreOp>(loc(node), value, alloca);
  }

  _functionSymbols[funcName] = getType(node->type()->name());
}

auto MLIRGenImpl::gen(const FunctionDecl *node,
                      mlir::FuncOp &func) -> void {
  _variableSymbols = {};
  gen(node->proto().get(), func);

  mlir::Value value;
  gen(node->body().get(), value);

  auto location = loc(node->body()->expression().get());
  if (value)
    _builder.create<ReturnOp>(location, value);
  else
    _builder.create<ReturnOp>(location, llvm::None);
}

auto MLIRGenImpl::gen(const StructDecl *node) -> void {
  auto &variables = node->variables();
  if (variables.size() == 0) {
    _typeSymbols[node->id()->name()] = _builder.getNoneType();
    return;
  }

  llvm::SmallVector<mlir::Type, 2> elementTypes;
  elementTypes.reserve(variables.size());
  for (auto &variable : variables) {
    mlir::Type type = getType(variable->varType()->name());
    elementTypes.push_back(type);
  }

  _typeSymbols[node->id()->name()] = StructType::get(elementTypes);
}

auto MLIRGenImpl::gen(const Expr *node, mlir::Value &value) -> void {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return gen(cast<UnitExpr>(node), value);
  case Expr::Expr_DecimalLiteral:
    return gen(cast<DecimalLiteralExpr>(node), value);
  case Expr::Expr_BoolLiteral:
    return gen(cast<BoolLiteralExpr>(node), value);
  case Expr::Expr_Call:
    return gen(cast<CallExpr>(node), value);
  case Expr::Expr_Variable:
    return gen(cast<VariableExpr>(node), value);
  case Expr::Expr_Binary:
    return gen(cast<BinaryExpr>(node), value);
  case Expr::Expr_If:
    return gen(cast<IfExpr>(node), value);
  case Expr::Expr_While:
    return gen(cast<WhileExpr>(node), value);
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto MLIRGenImpl::gen(const UnitExpr *node, mlir::Value &value) -> void {
  value = nullptr;
}

auto MLIRGenImpl::gen(const BlockExpr *node, mlir::Value &value) -> void {
  for (auto &expr : *node)
    gen(expr.get());
  gen(node->expression().get(), value);
}

auto MLIRGenImpl::gen(const IfExpr *node, mlir::Value &value) -> void {
  mlir::Value cond;
  gen(node->conditionExpr().get(), cond);

  auto thenBlock = node->thenBlock().get();
  auto thenExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    mlir::Value value;
    gen(thenBlock, value);
    builder.create<YieldOp>(loc(thenBlock->expression().get()), value);
  };

  auto elseBlock = node->elseBlock().get();
  auto elseExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    mlir::Value value;
    gen(elseBlock, value);
    builder.create<YieldOp>(loc(elseBlock->expression().get()), value);
  };
  auto ifOp = _builder.create<IfOp>(loc(node),
                                    getType(node->type()),
                                    cond,
                                    thenExprBuilder,
                                    elseExprBuilder);
  value = ifOp.getResult();
}

auto MLIRGenImpl::gen(const WhileExpr *node, mlir::Value &value) -> void {
  auto conditionExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    mlir::Value cond;
    gen(node->conditionExpr().get(), cond);
    builder.create<YieldOp>(loc(node->conditionExpr().get()), cond);
  };

  auto bodyBlock = node->bodyBlock().get();
  auto bodyExprBuilder = [&](mlir::OpBuilder &builder, mlir::Location) {
    mlir::Value value;
    gen(bodyBlock, value);
    builder.create<YieldOp>(loc(bodyBlock->expression().get()), llvm::None);
  };

  auto whileOp = _builder.create<WhileOp>(loc(node), conditionExprBuilder, bodyExprBuilder);
  value = nullptr;
}

auto MLIRGenImpl::gen(const CallExpr *node, mlir::Value &value) -> void {
  auto functionName = node->name();
  if (node->name() == builtins::print)
    return genPrint(node, value);

  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto &expr : *node) {
    mlir::Value value;
    gen(expr.get(), value);
    operands.push_back(value);
  }
  if (functionName == builtins::boolToUInt64) {
    value = _builder.create<CastOp>(loc(node),  operands.front());
  } else {
    auto op = _builder.create<CallOp>(loc(node), node->name(), operands, _functionSymbols[functionName]);
    value = node->type() == builtins::UnitType ? nullptr : op.getResult(0);
  }
}

auto MLIRGenImpl::genPrint(const CallExpr *node, mlir::Value &value) -> void {
  auto &expressions = node->expressions();
  mlir::Value operand;
  gen(expressions.front().get(), operand);

  value = _builder.create<PrintOp>(loc(node), operand);
}

auto MLIRGenImpl::gen(const VariableExpr *node, mlir::Value &value) -> void {
  auto address  = _variableSymbols[node->name()];
  value = _builder.create<mlir::LoadOp>(loc(node), address);
}

auto MLIRGenImpl::gen(const DecimalLiteralExpr *node, mlir::Value &value) -> void {
  value = _builder.create<ConstantOp>(loc(node), node->value());
}

auto MLIRGenImpl::gen(const BoolLiteralExpr *node, mlir::Value &value) -> void {
  value = _builder.create<ConstantOp>(loc(node), node->value());
}

auto MLIRGenImpl::gen(const BinaryExpr *node, mlir::Value &value) -> void {
  auto op = node->op();
  if (op == "=")
    return genAssign(node, value);
  else
    llvm_unreachable("Unexpected BinaryExpr operator");
}

auto MLIRGenImpl::genAssign(const BinaryExpr *node, mlir::Value &value) -> void {
  mlir::Value rhsValue;
  gen(node->rhs().get(), rhsValue);

  // TODO: handle struct access
  auto lhs = static_cast<VariableExpr *>(node->lhs().get());
  auto name = lhs->name();

  auto address = _variableSymbols[name];
  if (node->lhs()->type() != builtins::UnitType)
    _builder.create<mlir::StoreOp>(loc(node), rhsValue, address);

  value = nullptr;
}

auto MLIRGenImpl::gen(const Stat *node) -> void {
  switch (node->getKind()) {
  case Stat::Stat_VariableDecl:
    return gen(cast<VariableStat>(node));
  case Stat::Stat_Expression:
    return gen(cast<ExprStat>(node));
  default:
    llvm_unreachable("Unexpected statement");
  }
}

auto MLIRGenImpl::gen(const VariableStat *node) -> void {
  auto typeName = node->varType()->name();
  auto varName = node->variable()->name();
  auto alloca = createEntryBlockAlloca(getType(typeName), loc(node));
  _variableSymbols[varName] = alloca;

  mlir::Value initValue;
  gen(node->init().get(), initValue);

  if (typeName != builtins::UnitType)
    _builder.create<mlir::StoreOp>(loc(node), initValue, alloca);
}

auto MLIRGenImpl::gen(const ExprStat *node) -> void {
  mlir::Value value;
  gen(node->expression().get(), value);
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