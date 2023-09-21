//===--- CherryOps.cpp - Cherry dialect ops -------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/CherryOps.h"
#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::cherry;

auto ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       uint64_t value) -> void {
  auto dataAttribute = builder.getI64IntegerAttr(value);
  ConstantOp::build(builder, state, dataAttribute.getType(), dataAttribute);
}

auto ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       bool value) -> void {
  auto dataAttribute = builder.getBoolAttr(value);
  ConstantOp::build(builder, state, builder.getI1Type(), dataAttribute);
}

auto CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   StringRef callee, ArrayRef<mlir::Value> arguments,
                   ArrayRef<Type> results) -> void {
  state.addTypes(results);
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

auto IfOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state,
    mlir::Type resultType, mlir::Value cond,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> thenBuilder,
    function_ref<void(OpBuilder &, mlir::Location)> elseBuilder) -> void {
  state.addTypes(resultType);
  state.addOperands(cond);

  OpBuilder::InsertionGuard guard(builder);

  Region *thenRegion = state.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, state.location);

  Region *elseRegion = state.addRegion();
  builder.createBlock(elseRegion);
  elseBuilder(builder, state.location);
}

auto WhileOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> conditionBuilder,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder) -> void {
  state.addTypes({});
  state.addOperands({});

  OpBuilder::InsertionGuard guard(builder);

  Region *conditionRegion = state.addRegion();
  builder.createBlock(conditionRegion);
  conditionBuilder(builder, state.location);

  Region *bodyRegion = state.addRegion();
  builder.createBlock(bodyRegion);
  bodyBuilder(builder, state.location);
}

auto ArithmeticLogicOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state, mlir::Value lhs,
                              mlir::Value rhs, StringRef op, mlir::Type type)
    -> void {
  state.addTypes(type);
  state.addOperands({lhs, rhs});
  state.addAttribute("op", builder.getStringAttr(op));
}

auto StructReadOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value structValue, int64_t index) -> void {
  CherryStructType structTy =
      llvm::cast<CherryStructType>(structValue.getType());
  mlir::Type resultType = structTy.getTypes()[index];
  IntegerAttr indexAttr = builder.getI64IntegerAttr(index);
  build(builder, state, resultType, structValue, indexAttr);
}

auto StructWriteOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          mlir::Value structValue, ArrayRef<int64_t> indexes,
                          mlir::Value valueToStore) -> void {
  auto attrs = builder.getDenseI64ArrayAttr(indexes);
  build(builder, state, structValue.getType(), structValue, attrs,
        valueToStore);
}

#define GET_OP_CLASSES
#include "cherry/MLIRGen/IR/CherryOps.cpp.inc"
