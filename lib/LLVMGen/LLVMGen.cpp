//===--- LLVMGen.cpp - LLVM Generator -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "LLVMGen.h"
#include "cherry/AST/AST.h"
#include "cherry/Basic/Builtins.h"
#include "cherry/Basic/CherryResult.h"
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
using mlir::failure;
using mlir::success;

class LLVMGenImpl {
public:
  LLVMGenImpl(const llvm::SourceMgr &sourceManager, llvm::LLVMContext &context,
              bool enableOpt)
      : _sourceManager{sourceManager}, _context{context}, _builder{context},
        _enableOpt{enableOpt} {
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
  std::map<llvm::StringRef, llvm::AllocaInst*> _variableSymbols;
  std::map<llvm::StringRef, llvm::Type*> _typeSymbols;
  // Debug symbols
  std::unique_ptr<llvm::DIBuilder> _dBuilder;
  llvm::DICompileUnit *_dcu;
  std::vector<llvm::DIScope *> _dlexicalBlocks;
  std::map<llvm::StringRef, llvm::DIType*> _dTypeSymbols;

  // Declarations
  auto gen(const Decl *node) -> CherryResult;
  auto gen(const Prototype *node, llvm::Function *&func) -> CherryResult;
  auto gen(const FunctionDecl *node) -> CherryResult;

  // Expressions
  auto gen(const Expr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const BlockExpr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const CallExpr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const VariableExpr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const DecimalLiteralExpr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const BoolLiteralExpr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const BinaryExpr *node, llvm::Value *&value) -> CherryResult;
  auto genAssign(const BinaryExpr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const IfExpr *node, llvm::Value *&value) -> CherryResult;

  // Statements
  auto gen(const Stat *node) -> CherryResult;
  auto gen(const VariableStat *node) -> CherryResult;
  auto gen(const ExprStat *node) -> CherryResult;

  // Utility
  auto getType(llvm::StringRef name) -> llvm::Type* {
    if (name == builtins::UInt64Type) {
      return llvm::Type::getInt64Ty(_context);
    } else if (name == builtins::BoolType) {
      return llvm::Type::getInt1Ty(_context);
    } else {
      return _typeSymbols[name];
    }
  }

  auto getDebugType(llvm::StringRef name) -> llvm::DIType* {
    return _dTypeSymbols[name];
  }

  auto addBuiltins() -> void {
    if (_debugInfo) {
      auto dUInt64Ty = _dBuilder->createBasicType(builtins::UInt64Type, 64,
                                                  llvm::dwarf::DW_ATE_unsigned);
      auto dUInt1Ty = _dBuilder->createBasicType(builtins::BoolType, 1,
                                                 llvm::dwarf::DW_ATE_unsigned);
      _dTypeSymbols[builtins::UInt64Type] = dUInt64Ty;
      _dTypeSymbols[builtins::BoolType] = dUInt1Ty;
    }

    auto llvmI64Ty = llvm::IntegerType::get(_context, 64);
    auto llvmI32Ty = llvm::IntegerType::get(_context, 32);
    auto llvmI8Ty = llvm::IntegerType::get(_context, 8);
    auto llvmI8PtrTy = llvm::PointerType::get(llvmI8Ty, 0);

    llvm::Function *printfFunc;
    {
      // printf
      auto funcType = llvm::FunctionType::get(llvmI32Ty, {llvmI8PtrTy}, true);
      printfFunc = llvm::Function::Create(funcType,llvm::Function::ExternalLinkage,
                                          "printf",module.get());
    }

    {
      auto funcType = llvm::FunctionType::get(llvmI64Ty, {llvmI64Ty}, false);
      auto printFunc = llvm::Function::Create(funcType,llvm::Function::ExternalLinkage,
                                              builtins::print,module.get());

      llvm::BasicBlock *bb = llvm::BasicBlock::Create(_context, "entry", printFunc);
      _builder.SetInsertPoint(bb);

      auto formatSpecifier = _builder.CreateGlobalString("%llu\n");
      auto index = llvm::ConstantInt::get(_context, llvm::APInt(64, llvm::StringRef("0"), 10));
      auto formatSpecifierPtr = _builder.CreateGEP(formatSpecifier, {index, index});

      llvm::Value *arg = printFunc->args().begin();
      auto result32 = _builder.CreateCall(printfFunc, {formatSpecifierPtr, arg}, "printf");
      auto result64 = _builder.CreateIntCast(result32, llvmI64Ty, false);
      _builder.CreateRet(result64);
    }
  }

  auto createEntryBlockAlloca(llvm::Function *func,
                              llvm::StringRef varName,
                              llvm::Type *type) -> llvm::AllocaInst* {
    llvm::IRBuilder<> TmpB(&func->getEntryBlock(), func->getEntryBlock().begin());
    return TmpB.CreateAlloca(type, nullptr, varName);
  }


  auto emitLocation(const Node *node) -> void {
    if (_debugInfo) {
      if (!node)
        return _builder.SetCurrentDebugLocation(llvm::DebugLoc());
      auto scope = _dlexicalBlocks.empty() ? _dcu : _dlexicalBlocks.back();
      auto [line, col] = _sourceManager.getLineAndColumn(node->location());
      _builder.SetCurrentDebugLocation(llvm::DebugLoc::get(line, col, scope));
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

  for (auto &decl : node) {
    if (gen(decl.get()))
      return failure();
  }

  if (_debugInfo) {
    _dBuilder->finalize();
  }

  if (llvm::verifyModule(*module)) {
    llvm::errs() << "module verification error";
    return failure();
  }

  return success();
}

auto LLVMGenImpl::gen(const Decl *node) -> CherryResult {
  switch (node->getKind()) {
  case Decl::Decl_Function:
    return gen(cast<FunctionDecl>(node));
  default:
    llvm_unreachable("Unexpected declaration");
  }
}

auto LLVMGenImpl::gen(const Prototype *node, llvm::Function *&func) -> CherryResult {
  auto name = node->id()->name();
  auto resultType = node->type()->name();
  llvm::SmallVector<llvm::Type*, 3> argTypes;
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
  func = llvm::Function::Create(funcType,
                                llvm::Function::ExternalLinkage,
                                name,
                                module.get());

  llvm::BasicBlock *bb = llvm::BasicBlock::Create(_context, "entry", func);
  _builder.SetInsertPoint(bb);

  auto [line, col] = _sourceManager.getLineAndColumn(node->location());
  llvm::DIFile *unit;
  llvm::DISubprogram *sp;
  if (_debugInfo) {
    unit =
        _dBuilder->createFile(_dcu->getFilename(), _dcu->getDirectory());
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

      _dBuilder->insertDeclare(alloca, dparm, _dBuilder->createExpression(),
                               llvm::DebugLoc::get(line, 0, sp),
                               _builder.GetInsertBlock());
    }

    _builder.CreateStore(&arg, alloca);
    _variableSymbols[name] = alloca;
  }

  return success();
}

auto LLVMGenImpl::gen(const FunctionDecl *node) -> CherryResult {
  _variableSymbols = {};
  llvm::Function *func;
  if (gen(node->proto().get(), func))
    return failure();

  llvm::Value *value;
  if (gen(node->body().get(), value)) {
    func->eraseFromParent();
    return failure();
  }

  _builder.CreateRet(value);

  _pass->run(*func);

  if (_debugInfo) {
    _dlexicalBlocks.pop_back();
  }

  return success();
}

auto LLVMGenImpl::gen(const Expr *node, llvm::Value *&value) -> CherryResult {
  switch (node->getKind()) {
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
  default:
    llvm_unreachable("Unexpected expression");
  }
}

auto LLVMGenImpl::gen(const BlockExpr *node,
                      llvm::Value *&value) -> CherryResult {
  for (auto &expr : *node)
    if (gen(expr.get()))
      return failure();
  return gen(node->expression().get(), value);
}

auto LLVMGenImpl::gen(const CallExpr *node, llvm::Value *&value) -> CherryResult {
  emitLocation(node);
  llvm::SmallVector<llvm::Value*, 4> operands;
  for (auto &expr : *node) {
    llvm::Value *value;
    if (gen(expr.get(), value))
      return failure();
    operands.push_back(value);
  }
  auto functionName = node->name();
  if (functionName == builtins::boolToUInt64) {
    value = _builder.CreateIntCast(operands.front(),
                                   getType(builtins::UInt64Type), false);
  } else {
    llvm::Function *callee = module->getFunction(functionName);
    value = _builder.CreateCall(callee, operands, functionName);
  }

  return success();
}

auto LLVMGenImpl::gen(const VariableExpr *node, llvm::Value *&value) -> CherryResult {
  emitLocation(node);
  auto name = node->name();
  auto alloca = _variableSymbols[name];
  value = _builder.CreateLoad(alloca, name);
  return success();
}

auto LLVMGenImpl::gen(const DecimalLiteralExpr *node, llvm::Value *&value) -> CherryResult {
  emitLocation(node);
  value = llvm::ConstantInt::get(getType(builtins::UInt64Type), node->value());
  return success();
}

auto LLVMGenImpl::gen(const BoolLiteralExpr *node, llvm::Value *&value) -> CherryResult {
  emitLocation(node);
  value = llvm::ConstantInt::get(getType(builtins::BoolType), node->value());
  return success();
}

auto LLVMGenImpl::gen(const BinaryExpr *node, llvm::Value *&value) -> CherryResult {
  auto op = node->op();
  if (op == "=")
    return genAssign(node, value);
  else
    llvm_unreachable("Unexpected BinaryExpr operator");
}

auto LLVMGenImpl::genAssign(const BinaryExpr *node,
                            llvm::Value *&value) -> CherryResult {
  emitLocation(node);
  llvm::Value *rhsValue;
  if (gen(node->rhs().get(), rhsValue))
    return failure();

  // TODO: handle struct access
  auto lhs = static_cast<VariableExpr *>(node->lhs().get());
  auto name = lhs->name();
  auto alloca = _variableSymbols[name];
  _builder.CreateStore(rhsValue, alloca);
  value = rhsValue;
  return success();
}

auto LLVMGenImpl::gen(const IfExpr *node, llvm::Value *&value) -> CherryResult {
  llvm::Function *func = _builder.GetInsertBlock()->getParent();

  llvm::BasicBlock *thenBB = llvm::BasicBlock::Create(_context, "then");
  llvm::BasicBlock *elseBB = llvm::BasicBlock::Create(_context, "else");
  llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(_context, "ifcont");

  // Condition
  llvm::Value *conditionValue;
  if (gen(node->conditionExpr().get(), conditionValue))
    return failure();
  _builder.CreateCondBr(conditionValue, thenBB, elseBB);

  // Emit then block
  func->getBasicBlockList().push_back(thenBB);
  _builder.SetInsertPoint(thenBB);
  llvm::Value *thenValue;
  if (gen(node->thenBlock().get(), thenValue))
    return failure();
  _builder.CreateBr(mergeBB);
  thenBB = _builder.GetInsertBlock();

  // Emit else block
  func->getBasicBlockList().push_back(elseBB);
  _builder.SetInsertPoint(elseBB);
  llvm::Value *elseValue;
  if (gen(node->elseBlock().get(), elseValue))
    return failure();
  _builder.CreateBr(mergeBB);
  elseBB = _builder.GetInsertBlock();

  // Emit merge block
  func->getBasicBlockList().push_back(mergeBB);
  _builder.SetInsertPoint(mergeBB);

  llvm::PHINode *pn = _builder.CreatePHI(getType(node->type()), 2, "iftmp");
  pn->addIncoming(thenValue, thenBB);
  pn->addIncoming(elseValue, elseBB);
  value = pn;
}

auto LLVMGenImpl::gen(const Stat *node) -> CherryResult {
  switch (node->getKind()) {
  case Stat::Stat_VariableDecl:
    return gen(cast<VariableStat>(node));
  case Stat::Stat_Expression:
    return gen(cast<ExprStat>(node));
  default:
    llvm_unreachable("Unexpected statement");
  }
}

auto LLVMGenImpl::gen(const VariableStat *node) -> CherryResult {
  auto name = node->variable()->name();
  auto typeName = node->varType()->name();
  auto llvmType = getType(typeName);

  llvm::Value *initValue;
  if (gen(node->init().get(), initValue))
    return failure();

  auto func = _builder.GetInsertBlock()->getParent();
  auto alloca = createEntryBlockAlloca(func, name, llvmType);
  _variableSymbols[name] = alloca;
  emitLocation(node);
  _builder.CreateStore(initValue, alloca);
  return success();
}

auto LLVMGenImpl::gen(const ExprStat *node) -> CherryResult {
  llvm::Value *value;
  return gen(node->expression().get(), value);
}

namespace cherry {

auto llvmGen(const llvm::SourceMgr &sourceManager,
             llvm::LLVMContext &context,
             const Module &moduleAST,
             std::unique_ptr<llvm::Module> &module,
             bool enableOpt) -> CherryResult {
  auto generator = std::make_unique<LLVMGenImpl>(sourceManager, context, enableOpt);

  auto result =  generator->gen(moduleAST);
  module = std::move(generator->module);
  return result;
}

} // end namespace cherry
