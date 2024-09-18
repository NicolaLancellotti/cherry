//===--- LLVMGen.cpp - LLVM Generator -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/LLVMGen/LLVMGen.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/Builtins.h"
#include "cherry/Basic/CherryResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"
#include <map>
#include <memory>

namespace {
using namespace cherry;
using llvm::cast;
using llvm::failure;
using llvm::success;

class LLVMGenImpl {
public:
  LLVMGenImpl(const llvm::SourceMgr &sourceManager, llvm::LLVMContext &context,
              bool enableOpt)
      : _enableOpt{enableOpt},
        _sourceManager{sourceManager}, _context{context}, _builder{context} {
    _debugInfo = !enableOpt;
  }

  auto gen(const Module &node) -> CherryResult;

  std::unique_ptr<llvm::Module> module;

private:
  bool _debugInfo;
  bool _enableOpt;
  const llvm::SourceMgr &_sourceManager;
  llvm::LLVMContext &_context;
  llvm::IRBuilder<> _builder;
  std::unique_ptr<llvm::legacy::FunctionPassManager> _pass;
  std::map<llvm::StringRef, llvm::AllocaInst *> _variableSymbols;
  std::map<llvm::StringRef, llvm::Type *> _typeSymbols;

  llvm::Value *structAddress = nullptr;
  const std::string tmpExpression = "0tmp";

  // Debug symbols
  std::unique_ptr<llvm::DIBuilder> _dBuilder;
  llvm::DICompileUnit *_dcu;
  std::vector<llvm::DIScope *> _dlexicalBlocks;
  std::map<llvm::StringRef, llvm::DIType *> _dTypeSymbols;

  // Declarations
  auto gen(const Decl *node) -> void;
  auto gen(const Prototype *node) -> llvm::Function *;
  auto gen(const FunctionDecl *node) -> void;
  auto gen(const StructDecl *node) -> void;

  // Expressions
  auto gen(const Expr *node) -> llvm::Value *;
  auto gen(const UnitExpr *node) -> llvm::Value *;
  auto gen(const BlockExpr *node) -> llvm::Value *;
  auto gen(const CallExpr *node) -> llvm::Value *;
  auto genStructInitializer(const CallExpr *node) -> llvm::Value *;
  auto gen(const VariableExpr *node) -> llvm::Value *;
  auto gen(const DecimalLiteralExpr *node) -> llvm::Value *;
  auto gen(const BoolLiteralExpr *node) -> llvm::Value *;
  auto gen(const BinaryExpr *node) -> llvm::Value *;
  auto genAssignOp(const BinaryExpr *node) -> llvm::Value *;
  auto genStructReadOp(const BinaryExpr *node) -> llvm::Value *;
  auto genStructAddress(const BinaryExpr *node)
      -> std::pair<llvm::Value *, llvm::Type *>;
  auto gen(const IfExpr *node) -> llvm::Value *;
  auto gen(const WhileExpr *node) -> llvm::Value *;

  // Statements
  auto gen(const Stat *node) -> void;
  auto gen(const VariableStat *node) -> void;
  auto gen(const ExprStat *node) -> void;

  // Utility
  auto getConstantInt(int bits, int value) -> llvm::ConstantInt * {
    return llvm::ConstantInt::get(_context, llvm::APInt(bits, value, true));
  }

  auto getUnit() -> llvm::Value * {
    return llvm::UndefValue::get(getType(builtins::UnitType));
  }

  auto getType(llvm::StringRef name) -> llvm::Type * {
    if (name == builtins::UnitType) {
      return llvm::Type::getVoidTy(_context);
    } else if (name == builtins::UInt64Type) {
      return llvm::Type::getInt64Ty(_context);
    } else if (name == builtins::BoolType) {
      return llvm::Type::getInt1Ty(_context);
    } else {
      return _typeSymbols[name];
    }
  }

  auto addType(const StructDecl *node) {
    auto name = node->id()->name();
    llvm::SmallVector<llvm::Type *, 3> types;
    for (auto &field : *node)
      types.push_back(getType(field->varType()->name()));
    _typeSymbols[name] = llvm::StructType::create(_context, types, name);
  }

  auto getDebugType(llvm::StringRef name) -> llvm::DIType * {
    return _dTypeSymbols[name];
  }

  auto addBuiltins() -> void {
    if (_debugInfo) {
      auto dUInt64Ty = _dBuilder->createBasicType(builtins::UInt64Type, 64,
                                                  llvm::dwarf::DW_ATE_unsigned);
      auto dUInt1Ty = _dBuilder->createBasicType(builtins::BoolType, 1,
                                                 llvm::dwarf::DW_ATE_unsigned);
      auto dUnitTy = _dBuilder->createBasicType(builtins::UnitType, 0,
                                                llvm::dwarf::DW_ATE_unsigned);
      _dTypeSymbols[builtins::UInt64Type] = dUInt64Ty;
      _dTypeSymbols[builtins::BoolType] = dUInt1Ty;
      _dTypeSymbols[builtins::UnitType] = dUnitTy;
    }

    auto llvmI64Ty = llvm::IntegerType::get(_context, 64);
    auto llvmI32Ty = llvm::IntegerType::get(_context, 32);
    auto llvmI8Ty = llvm::IntegerType::get(_context, 8);
    auto llvmI8PtrTy = llvm::PointerType::get(llvmI8Ty, 0);

    llvm::Function *printfFunc;
    {
      // printf
      auto funcType = llvm::FunctionType::get(llvmI32Ty, {llvmI8PtrTy}, true);
      printfFunc = llvm::Function::Create(
          funcType, llvm::Function::ExternalLinkage, "printf", module.get());
    }

    {
      auto funcType = llvm::FunctionType::get(llvmI64Ty, {llvmI64Ty}, false);
      auto printFunc =
          llvm::Function::Create(funcType, llvm::Function::ExternalLinkage,
                                 builtins::print, module.get());

      auto *bb = llvm::BasicBlock::Create(_context, "entry", printFunc);
      _builder.SetInsertPoint(bb);

      auto formatSpecifier = _builder.CreateGlobalString("%llu\n");
      auto index0 = getConstantInt(64, 0);
      auto formatSpecifierPtr =
          _builder.CreateGEP(llvmI8PtrTy, formatSpecifier, {index0});

      auto *arg = printFunc->args().begin();
      auto result32 =
          _builder.CreateCall(printfFunc, {formatSpecifierPtr, arg}, "printf");
      auto result64 = _builder.CreateIntCast(result32, llvmI64Ty, false);
      _builder.CreateRet(result64);
    }
  }

  auto createEntryBlockAlloca(llvm::Function *func, llvm::StringRef varName,
                              llvm::Type *type) -> llvm::AllocaInst * {
    if (type == getType(builtins::UnitType))
      return nullptr;
    llvm::IRBuilder<> TmpB(&func->getEntryBlock(),
                           func->getEntryBlock().begin());
    return TmpB.CreateAlloca(type, nullptr, varName);
  }

  auto emitLocation(const Node *node) -> void {
    if (_debugInfo) {
      if (!node)
        return _builder.SetCurrentDebugLocation(llvm::DebugLoc());
      auto scope = _dlexicalBlocks.empty() ? _dcu : _dlexicalBlocks.back();
      auto [line, col] = _sourceManager.getLineAndColumn(node->location());
      _builder.SetCurrentDebugLocation(
          llvm::DILocation::get(scope->getContext(), line, col, scope));
    }
  }
};

} // end namespace

auto LLVMGenImpl::gen(const Module &node) -> CherryResult {
  module = std::make_unique<llvm::Module>("cherry module", _context);

  _pass = std::make_unique<llvm::legacy::FunctionPassManager>(module.get());
  if (_enableOpt) {
    // Promote allocas to registers.
    _pass->add(llvm::createPromoteMemoryToRegisterPass());
    // Do simple "peephole" optimizations and bit-twiddling optzns.
    _pass->add(llvm::createInstructionCombiningPass());
    // Reassociate expressions.
    _pass->add(llvm::createReassociatePass());
    // Eliminate Common SubExpressions.
    _pass->add(llvm::createGVNPass());
    // Simplify the control flow graph (deleting unreachable blocks, etc).
    _pass->add(llvm::createCFGSimplificationPass());

    _pass->doInitialization();
  }

  if (_debugInfo) {
    auto fileName =
        _sourceManager.getMemoryBuffer(_sourceManager.getMainFileID())
            ->getBufferIdentifier();
    _dBuilder = std::make_unique<llvm::DIBuilder>(*module);
    _dcu = _dBuilder->createCompileUnit(llvm::dwarf::DW_LANG_C,
                                        _dBuilder->createFile(fileName, "."),
                                        "Cherry Compiler", 0, "", 0);
  }

  addBuiltins();

  for (auto &decl : node)
    gen(decl.get());

  if (_debugInfo)
    _dBuilder->finalize();

  if (llvm::verifyModule(*module, &llvm::errs())) {
    llvm::errs() << "module verification error";
    return failure();
  }

  return success();
}

auto LLVMGenImpl::gen(const Decl *node) -> void {
  switch (node->getKind()) {
  case Decl::Decl_Function:
    return gen(cast<FunctionDecl>(node));
  case Decl::Decl_Struct:
    return gen(cast<StructDecl>(node));
  }
}

auto LLVMGenImpl::gen(const Prototype *node) -> llvm::Function * {

  auto name = node->id()->name();
  auto resultType = node->type()->name();
  llvm::SmallVector<llvm::Type *, 3> argTypes;
  argTypes.reserve(node->parameters().size());

  llvm::SmallVector<llvm::Metadata *, 8> debugTypes;
  if (_debugInfo) {
    debugTypes.push_back(getDebugType(resultType)); // result
  }

  for (auto &param : node->parameters()) {
    auto typeName = param->varType()->name();
    argTypes.push_back(getType(typeName));
    if (_debugInfo) {
      debugTypes.push_back(getDebugType(typeName));
    }
  }

  auto llvmResultType = getType(resultType);
  auto funcType = llvm::FunctionType::get(llvmResultType, argTypes, false);
  auto func = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage,
                                     name, module.get());

  auto *bb = llvm::BasicBlock::Create(_context, "entry", func);
  _builder.SetInsertPoint(bb);

  auto [line, col] = _sourceManager.getLineAndColumn(node->location());
  llvm::DIFile *unit;
  llvm::DISubprogram *sp;
  if (_debugInfo) {
    unit = _dBuilder->createFile(_dcu->getFilename(), _dcu->getDirectory());
    auto subroutineType = _dBuilder->createSubroutineType(
        _dBuilder->getOrCreateTypeArray(debugTypes));
    sp = _dBuilder->createFunction(
        unit, name, llvm::StringRef(), unit, line, subroutineType, line,
        llvm::DINode::FlagPrototyped, llvm::DISubprogram::SPFlagDefinition);
    func->setSubprogram(sp);
    _dlexicalBlocks.push_back(sp);
    emitLocation(nullptr);
  }

  unsigned index = 0;
  for (const auto &param_arg : llvm::zip(node->parameters(), func->args())) {
    auto &param = std::get<0>(param_arg);
    auto &arg = std::get<1>(param_arg);
    auto name = param->variable()->name();
    arg.setName(name);

    auto alloca = createEntryBlockAlloca(func, name, arg.getType());
    if (_debugInfo) {
      auto dparm = _dBuilder->createParameterVariable(
          sp, name, ++index, unit, line, getDebugType(param->varType()->name()),
          true);
      _dBuilder->insertDeclare(
          alloca, dparm, _dBuilder->createExpression(),
          llvm::DILocation::get(sp->getContext(), line, 0, sp),
          _builder.GetInsertBlock());
    }
    _builder.CreateStore(&arg, alloca);
    _variableSymbols[name] = alloca;
  }
  return func;
}

auto LLVMGenImpl::gen(const FunctionDecl *node) -> void {
  _variableSymbols = {};
  auto *func = gen(node->proto().get());
  auto *value = gen(node->body().get());

  if (node->proto()->type()->name() == builtins::UnitType) {
    _builder.CreateRetVoid();
  } else {
    if (!value) {
      auto llvmType = getType(node->proto()->type()->name());
      auto *structType = static_cast<llvm::StructType *>(llvmType);
      auto index0 = getConstantInt(32, 0);
      auto alloca = _variableSymbols[tmpExpression];
      auto address = _builder.CreateGEP(structType, alloca, index0);
      value = _builder.CreateLoad(structType, address, "tmp");
    }
    _builder.CreateRet(value);
  }

  _pass->run(*func);

  if (_debugInfo)
    _dlexicalBlocks.pop_back();
}

auto LLVMGenImpl::gen(const StructDecl *node) -> void { addType(node); }

auto LLVMGenImpl::gen(const Expr *node) -> llvm::Value * {
  switch (node->getKind()) {
  case Expr::Expr_Unit:
    return gen(cast<UnitExpr>(node));
  case Expr::Expr_DecimalLiteral:
    return gen(cast<DecimalLiteralExpr>(node));
  case Expr::Expr_BoolLiteral:
    return gen(cast<BoolLiteralExpr>(node));
  case Expr::Expr_Call:
    return gen(cast<CallExpr>(node));
  case Expr::Expr_Variable:
    return gen(cast<VariableExpr>(node));
  case Expr::Expr_Binary:
    return gen(cast<BinaryExpr>(node));
  case Expr::Expr_If:
    return gen(cast<IfExpr>(node));
  case Expr::Expr_While:
    return gen(cast<WhileExpr>(node));
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto LLVMGenImpl::gen(const UnitExpr *node) -> llvm::Value * {
  emitLocation(node);
  return getUnit();
}

auto LLVMGenImpl::gen(const BlockExpr *node) -> llvm::Value * {
  for (auto &expr : *node)
    gen(expr.get());
  return gen(node->expression().get());
}

auto LLVMGenImpl::gen(const CallExpr *node) -> llvm::Value * {
  if (getType(node->name()))
    return genStructInitializer(node);

  emitLocation(node);
  llvm::SmallVector<llvm::Value *, 4> operands;
  for (auto &expr : *node) {
    auto *value = gen(expr.get());
    operands.push_back(value);
  }
  auto functionName = node->name();
  if (functionName == builtins::boolToUInt64) {
    return _builder.CreateIntCast(operands.front(),
                                  getType(builtins::UInt64Type), false);
  } else {
    auto *callee = module->getFunction(functionName);
    return _builder.CreateCall(callee, operands);
  }
}

auto LLVMGenImpl::genStructInitializer(const CallExpr *node) -> llvm::Value * {
  emitLocation(node);
  auto llvmType = getType(node->name());
  auto *structType = static_cast<llvm::StructType *>(llvmType);
  auto index0 = getConstantInt(32, 0);

  if (!structAddress) {
    auto func = _builder.GetInsertBlock()->getParent();
    auto *alloca = createEntryBlockAlloca(func, "tmp", llvmType);
    _variableSymbols[tmpExpression] = alloca;
    structAddress = alloca;
  }
  auto address = structAddress;

  auto index = 0;
  for (auto &expr : *node) {
    auto fieldIndex = getConstantInt(32, index++);
    auto fieldAddress =
        _builder.CreateGEP(structType, address, {index0, fieldIndex});
    structAddress = fieldAddress;
    if (llvm::Value *value = gen(expr.get()))
      _builder.CreateStore(value, fieldAddress);
  }
  structAddress = nullptr;
  return nullptr;
}

auto LLVMGenImpl::gen(const VariableExpr *node) -> llvm::Value * {
  emitLocation(node);
  auto name = node->name();
  if (auto alloca = _variableSymbols[name])
    return _builder.CreateLoad(alloca->getAllocatedType(), alloca, name);
  else
    return getUnit();
}

auto LLVMGenImpl::gen(const DecimalLiteralExpr *node) -> llvm::Value * {
  emitLocation(node);
  return llvm::ConstantInt::get(getType(builtins::UInt64Type), node->value());
}

auto LLVMGenImpl::gen(const BoolLiteralExpr *node) -> llvm::Value * {
  emitLocation(node);
  return llvm::ConstantInt::get(getType(builtins::BoolType), node->value());
}

auto LLVMGenImpl::gen(const BinaryExpr *node) -> llvm::Value * {
  using Operator = BinaryExpr::Operator;
  auto op = node->opEnum();
  switch (op) {
  case Operator::Assign:
    return genAssignOp(node);
  case Operator::StructRead:
    return genStructReadOp(node);
  default:
    break;
  }

  auto lhs = gen(node->lhs().get());
  auto rhs = gen(node->rhs().get());
  switch (op) {
  case Operator::Add:
    return _builder.CreateAdd(lhs, rhs);
  case Operator::Mul:
    return _builder.CreateMul(lhs, rhs);
  case Operator::Diff:
    return _builder.CreateSub(lhs, rhs);
  case Operator::Div:
    return _builder.CreateUDiv(lhs, rhs);
  case Operator::Rem:
    return _builder.CreateURem(lhs, rhs);
  case Operator::And:
    return _builder.CreateAnd(lhs, rhs);
  case Operator::Or:
    return _builder.CreateOr(lhs, rhs);
  case Operator::LT:
    return _builder.CreateICmpULT(lhs, rhs);
  case Operator::LE:
    return _builder.CreateICmpULE(lhs, rhs);
  case Operator::GT:
    return _builder.CreateICmpUGT(lhs, rhs);
  case Operator::GE:
    return _builder.CreateICmpUGE(lhs, rhs);
  case Operator::EQ:
    return _builder.CreateICmpEQ(lhs, rhs);
  case Operator::NEQ:
    return _builder.CreateICmpNE(lhs, rhs);
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto LLVMGenImpl::genAssignOp(const BinaryExpr *node) -> llvm::Value * {
  emitLocation(node);
  llvm::Value *address;
  llvm::TypeSwitch<const Expr *>(node->lhs().get())
      .Case<VariableExpr>([&](const auto *node) {
        address = structAddress = _variableSymbols[node->name()];
      })
      .Case<BinaryExpr>([&](const auto *node) {
        address = structAddress = genStructAddress(node).first;
      })
      .Default(
          [&](const Expr *) { llvm_unreachable("Unexpected expression"); });

  auto value = gen(node->rhs().get());
  if (value && node->lhs()->type() != builtins::UnitType)
    _builder.CreateStore(value, address);
  structAddress = nullptr;
  return getUnit();
}

auto LLVMGenImpl::genStructReadOp(const BinaryExpr *node) -> llvm::Value * {
  emitLocation(node);
  auto [address, type] = genStructAddress(node);
  return _builder.CreateLoad(type, address);
}

auto LLVMGenImpl::genStructAddress(const BinaryExpr *node)
    -> std::pair<llvm::Value *, llvm::Type *> {
  emitLocation(node);
  auto lhs = node->lhs().get();
  auto *structType = static_cast<llvm::StructType *>(getType(lhs->type()));
  auto index0 = getConstantInt(32, 0);
  auto fieldIndex = getConstantInt(32, node->index());
  llvm::Value *address;
  llvm::TypeSwitch<const Expr *>(lhs)
      .Case<VariableExpr>(
          [&](auto *node) { address = _variableSymbols[node->name()]; })
      .Case<BinaryExpr>(
          [&](const auto *node) { address = genStructAddress(node).first; })
      .Default(
          [&](const Expr *) { llvm_unreachable("Unexpected expression"); });

  auto type = structType->getTypeAtIndex(node->index());
  return std::make_pair(
      _builder.CreateGEP(structType, address, {index0, fieldIndex}), type);
}

auto LLVMGenImpl::gen(const IfExpr *node) -> llvm::Value * {
  emitLocation(node);
  auto *func = _builder.GetInsertBlock()->getParent();

  auto *thenBB = llvm::BasicBlock::Create(_context, "then");
  auto *elseBB = llvm::BasicBlock::Create(_context, "else");
  auto *mergeBB = llvm::BasicBlock::Create(_context, "ifcont");

  // Emit condition
  auto *conditionValue = gen(node->conditionExpr().get());
  _builder.CreateCondBr(conditionValue, thenBB, elseBB);

  // Emit then block
  func->insert(func->end(), thenBB);
  _builder.SetInsertPoint(thenBB);
  auto *thenValue = gen(node->thenBlock().get());
  _builder.CreateBr(mergeBB);
  thenBB = _builder.GetInsertBlock();

  // Emit else block
  func->insert(func->end(), elseBB);
  _builder.SetInsertPoint(elseBB);
  auto *elseValue = gen(node->elseBlock().get());
  _builder.CreateBr(mergeBB);
  elseBB = _builder.GetInsertBlock();

  // Emit merge block
  func->insert(func->end(), mergeBB);
  _builder.SetInsertPoint(mergeBB);

  auto *pn = _builder.CreatePHI(getType(node->type()), 2, "iftmp");
  pn->addIncoming(thenValue, thenBB);
  pn->addIncoming(elseValue, elseBB);
  return pn;
}

auto LLVMGenImpl::gen(const WhileExpr *node) -> llvm::Value * {
  emitLocation(node);
  auto *func = _builder.GetInsertBlock()->getParent();
  auto *initial = _builder.GetInsertBlock();
  auto *conditionBB = llvm::BasicBlock::Create(_context, "condition");
  auto *loopBB = llvm::BasicBlock::Create(_context, "loop");
  auto *afterLoopBB = llvm::BasicBlock::Create(_context, "afterloop");

  _builder.SetInsertPoint(initial);
  _builder.CreateBr(conditionBB);

  // Emit condition
  func->insert(func->end(), conditionBB);
  _builder.SetInsertPoint(conditionBB);
  llvm::Value *conditionValue = gen(node->conditionExpr().get());
  _builder.CreateCondBr(conditionValue, loopBB, afterLoopBB);

  // Emit body block
  func->insert(func->end(), loopBB);
  _builder.SetInsertPoint(loopBB);
  gen(node->bodyBlock().get());
  _builder.CreateBr(conditionBB);

  // Emit after loop block
  func->insert(func->end(), afterLoopBB);
  _builder.SetInsertPoint(afterLoopBB);

  return nullptr;
}

auto LLVMGenImpl::gen(const Stat *node) -> void {
  switch (node->getKind()) {
  case Stat::Stat_VariableDecl:
    return gen(cast<VariableStat>(node));
  case Stat::Stat_Expression:
    return gen(cast<ExprStat>(node));
  }
}

auto LLVMGenImpl::gen(const VariableStat *node) -> void {
  auto name = node->variable()->name();
  auto typeName = node->varType()->name();
  auto llvmType = getType(typeName);

  auto func = _builder.GetInsertBlock()->getParent();
  auto alloca = createEntryBlockAlloca(func, name, llvmType);
  structAddress = alloca;
  _variableSymbols[name] = alloca;

  auto *initValue = gen(node->init().get());
  structAddress = nullptr;

  emitLocation(node);
  if (initValue && typeName != builtins::UnitType)
    _builder.CreateStore(initValue, alloca);
}

auto LLVMGenImpl::gen(const ExprStat *node) -> void {
  gen(node->expression().get());
}

namespace cherry {

auto llvmGen(const llvm::SourceMgr &sourceManager, llvm::LLVMContext &context,
             const Module &moduleAST, std::unique_ptr<llvm::Module> &module,
             bool enableOpt) -> CherryResult {
  auto generator =
      std::make_unique<LLVMGenImpl>(sourceManager, context, enableOpt);
  auto result = generator->gen(moduleAST);
  module = std::move(generator->module);
  return result;
}

} // end namespace cherry
