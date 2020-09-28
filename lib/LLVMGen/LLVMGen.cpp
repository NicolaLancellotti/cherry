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

namespace {
using namespace cherry;
using llvm::cast;
using mlir::failure;
using mlir::success;

class LLVMGenImpl {
public:
  LLVMGenImpl(llvm::LLVMContext &context)
      : _context{context}, _builder{context} {}

  auto gen(const Module &node) -> CherryResult;

  std::unique_ptr<llvm::Module> module;
private:
  llvm::LLVMContext &_context;
  llvm::IRBuilder<> _builder;
  std::map<llvm::StringRef, llvm::Value*> _variableSymbols;
  std::map<llvm::StringRef, llvm::Type*> _typeSymbols;

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

  auto addBuiltins() -> void {
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
};

} // end namespace

auto LLVMGenImpl::gen(const Module &node) -> CherryResult {
  module = std::make_unique<llvm::Module>("cherry module", _context);
  addBuiltins();

  for (auto &decl : node) {
    if (gen(decl.get()))
      return failure();
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
  llvm::SmallVector<llvm::Type*, 3> arg_types;
  arg_types.reserve(node->parameters().size());
  for (auto &param : node->parameters())
    arg_types.push_back(getType(param->type()->name()));

  auto result_type = getType(types::UInt64Type);

  auto funcType = llvm::FunctionType::get(result_type, arg_types, false);
  func = llvm::Function::Create(funcType,
                                llvm::Function::ExternalLinkage,
                                node->id()->name(),
                                module.get());

  llvm::BasicBlock *bb = llvm::BasicBlock::Create(_context, "entry", func);
  _builder.SetInsertPoint(bb);

  for (const auto &param_arg : llvm::zip(node->parameters(), func->args())) {
    auto &param = std::get<0>(param_arg);
    auto &arg = std::get<1>(param_arg);
    auto name = param->variable()->name();
    arg.setName(name);

    auto alloca = createEntryBlockAlloca(func, name, arg.getType());
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
  auto name = node->name();
  auto alloca = _variableSymbols[name];
  value = _builder.CreateLoad(alloca, name);
  return success();
}

auto LLVMGenImpl::gen(const DecimalExpr *node, llvm::Value *&value) -> CherryResult {
  value = llvm::ConstantInt::get(getType(types::UInt64Type), node->value());
  return success();
}

namespace cherry {

auto llvmGen(llvm::LLVMContext &context,
             const Module &moduleAST) -> std::unique_ptr<llvm::Module> {
  auto generator = std::make_unique<LLVMGenImpl>(context);
  return generator->gen(moduleAST) ? nullptr : std::move(generator->module);
}

} // end namespace cherry
