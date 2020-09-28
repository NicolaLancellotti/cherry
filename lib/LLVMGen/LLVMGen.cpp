//===--- LLVMGen.cpp - LLVM Generator -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "LLVMGen.h"
#include "cherry/Basic/CherryTypes.h"
#include "cherry/Basic/CherryResult.h"
#include "cherry/AST/AST.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <memory>

#define CHERRY_DEBUG_INFO

namespace {
using namespace cherry;
using llvm::cast;
using mlir::failure;
using mlir::success;

class LLVMGenImpl {
public:
  LLVMGenImpl(const llvm::SourceMgr &sourceManager, llvm::LLVMContext &context)
      : _sourceManager{sourceManager}, _context{context}, _builder{context} {}

  auto gen(const Module &node) -> CherryResult;

  std::unique_ptr<llvm::Module> module;
private:
  const llvm::SourceMgr &_sourceManager;
  llvm::LLVMContext &_context;
  llvm::IRBuilder<> _builder;
  std::map<llvm::StringRef, llvm::Value*> _variableSymbols;
  std::map<llvm::StringRef, llvm::Type*> _typeSymbols;
#ifdef CHERRY_DEBUG_INFO
  std::unique_ptr<llvm::DIBuilder> _dBuilder;
  llvm::DICompileUnit *_dcu;
  std::vector<llvm::DIScope *> _dlexicalBlocks;
  std::map<llvm::StringRef, llvm::DIType*> _dTypeSymbols;
#endif

  // Declarations
  auto gen(const Decl *node) -> CherryResult;
  auto gen(const Prototype *node, llvm::Function *&func) -> CherryResult;
  auto gen(const FunctionDecl *node) -> CherryResult;

  // Expressions
  auto gen(const Expr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const CallExpr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const VariableExpr *node, llvm::Value *&value) -> CherryResult;
  auto gen(const DecimalExpr *node, llvm::Value *&value) -> CherryResult;

  // Utility
  auto getType(llvm::StringRef name) -> llvm::Type* {
    if (name == types::UInt64Type) {
      return llvm::Type::getInt64Ty(_context);
    } else {
      return _typeSymbols[name];
    }
  }

#ifdef CHERRY_DEBUG_INFO
  auto getDebugType(llvm::StringRef name) -> llvm::DIType* {
    return _dTypeSymbols[name];
  }
#endif

  auto addBuiltins() -> void {
#ifdef CHERRY_DEBUG_INFO
    auto dUInt64Ty = _dBuilder->createBasicType(types::UInt64Type, 64,
                                                llvm::dwarf::DW_ATE_unsigned);
    _dTypeSymbols[types::UInt64Type] = dUInt64Ty;
#endif

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
                                              "print",module.get());

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
#ifdef CHERRY_DEBUG_INFO
    if (!node)
      return _builder.SetCurrentDebugLocation(llvm::DebugLoc());
    auto scope = _dlexicalBlocks.empty() ? _dcu : _dlexicalBlocks.back();
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    _builder.SetCurrentDebugLocation(llvm::DebugLoc::get(line, col, scope));
#endif
  }

};

} // end namespace

auto LLVMGenImpl::gen(const Module &node) -> CherryResult {
  module = std::make_unique<llvm::Module>("cherry module", _context);
#ifdef CHERRY_DEBUG_INFO
  auto fileName = _sourceManager.getMemoryBuffer(_sourceManager.getMainFileID())
      ->getBufferIdentifier();
  _dBuilder = std::make_unique<llvm::DIBuilder>(*module);
  _dcu = _dBuilder->createCompileUnit(
      llvm::dwarf::DW_LANG_C,
      _dBuilder->createFile(fileName, "."),
      "Cherry Compiler", 0, "", 0);
#endif

  addBuiltins();

  for (auto &decl : node) {
    if (gen(decl.get()))
      return failure();
  }

#ifdef CHERRY_DEBUG_INFO
  _dBuilder->finalize();
#endif

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
  llvm::SmallVector<llvm::Type*, 3> argTypes;
  argTypes.reserve(node->parameters().size());

  llvm::SmallVector<llvm::Metadata *, 8> debugTypes;
#ifdef CHERRY_DEBUG_INFO
  debugTypes.push_back(getDebugType(types::UInt64Type)); // result
#endif

  for (auto &param : node->parameters()) {
    auto typeName = param->type()->name();
    argTypes.push_back(getType(typeName));
#ifdef CHERRY_DEBUG_INFO
    debugTypes.push_back(getDebugType(typeName));
#endif
  }

  auto resultType = getType(types::UInt64Type);
  auto funcType = llvm::FunctionType::get(resultType, argTypes, false);
  func = llvm::Function::Create(funcType,
                                llvm::Function::ExternalLinkage,
                                name,
                                module.get());

  llvm::BasicBlock *bb = llvm::BasicBlock::Create(_context, "entry", func);
  _builder.SetInsertPoint(bb);

#ifdef CHERRY_DEBUG_INFO
  auto [line, col] = _sourceManager.getLineAndColumn(node->location());
  auto unit = _dBuilder->createFile(_dcu->getFilename(), _dcu->getDirectory());
  auto subroutineType = _dBuilder->createSubroutineType(
      _dBuilder->getOrCreateTypeArray(debugTypes));
  auto sp = _dBuilder->createFunction(
      unit, name, llvm::StringRef(), unit, line, subroutineType, line,
      llvm::DINode::FlagPrototyped, llvm::DISubprogram::SPFlagDefinition);
  func->setSubprogram(sp);
  _dlexicalBlocks.push_back(sp);
  emitLocation(nullptr);
#endif

  unsigned index = 0;
  for (const auto &param_arg : llvm::zip(node->parameters(), func->args())) {
    auto &param = std::get<0>(param_arg);
    auto &arg = std::get<1>(param_arg);
    auto name = param->variable()->name();
    arg.setName(name);

    auto alloca = createEntryBlockAlloca(func, name, arg.getType());
#ifdef CHERRY_DEBUG_INFO
    auto dparm = _dBuilder->createParameterVariable(
        sp, name, ++index, unit, line, getDebugType(param->type()->name()),
        true);

    _dBuilder->insertDeclare(alloca, dparm, _dBuilder->createExpression(),
                             llvm::DebugLoc::get(line, 0, sp),
                             _builder.GetInsertBlock());
#endif

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

  for (auto &expr : *node) {
    llvm::Value *value;
    if (gen(expr.get(), value)) {
      func->eraseFromParent();
      return failure();
    }
  }

  auto constant0 = llvm::ConstantInt::get(getType(types::UInt64Type), 0);
  _builder.CreateRet(constant0);

#ifdef CHERRY_DEBUG_INFO
  _dlexicalBlocks.pop_back();
#endif

  return success();
}

auto LLVMGenImpl::gen(const Expr *node, llvm::Value *&value) -> CherryResult {
  switch (node->getKind()) {
  case Expr::Expr_Decimal:
    return gen(cast<DecimalExpr>(node), value);
  case Expr::Expr_Call:
    return gen(cast<CallExpr>(node), value);
  case Expr::Expr_Variable:
    return gen(cast<VariableExpr>(node), value);
  default:
    llvm_unreachable("Unexpected expression");
  }
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
  llvm::Function *callee = module->getFunction(functionName);
  value = _builder.CreateCall(callee, operands, functionName);
  return success();
}

auto LLVMGenImpl::gen(const VariableExpr *node, llvm::Value *&value) -> CherryResult {
  emitLocation(node);
  auto name = node->name();
  auto alloca = _variableSymbols[name];
  value = _builder.CreateLoad(alloca, name);
  return success();
}

auto LLVMGenImpl::gen(const DecimalExpr *node, llvm::Value *&value) -> CherryResult {
  emitLocation(node);
  value = llvm::ConstantInt::get(getType(types::UInt64Type), node->value());
  return success();
}

namespace cherry {

auto llvmGen(const llvm::SourceMgr &sourceManager,
             llvm::LLVMContext &context,
             const Module &moduleAST,
             std::unique_ptr<llvm::Module> &module) -> CherryResult {
  auto generator = std::make_unique<LLVMGenImpl>(sourceManager, context);

  auto result =  generator->gen(moduleAST);
  module = std::move(generator->module);
  return result;
}

} // end namespace cherry
