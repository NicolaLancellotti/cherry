//===--- CherryOps.cpp - Cherry dialect ops -------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/CherryOps.h"
#include "cherry/MLIRGen/CherryDialect.h"
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
  ConstantOp::build(builder, state, dataAttribute.getType(), dataAttribute);
}

auto CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   StringRef callee, ArrayRef<mlir::Value> arguments,
                   ArrayRef<Type> results) -> void {
  state.addTypes(results);
  state.addOperands(arguments);
  state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

auto PrintOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value argument) -> void {
  auto dataType = builder.getI64Type();
  state.addTypes(dataType);
  state.addOperands({argument});
}

auto CastOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value argument) -> void {
  auto dataType = builder.getI64Type();
  state.addTypes(dataType);
  state.addOperands({argument});
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
  state.addTypes(llvm::None);
  state.addOperands(llvm::None);

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
  ;
  state.addOperands({lhs, rhs});

  auto dataAttribute = StringAttr::get(op, builder.getContext());
  state.addAttribute("op", dataAttribute);
}

#define GET_OP_CLASSES
#include "cherry/MLIRGen/CherryOps.cpp.inc"