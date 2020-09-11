//===--- CherryOps.cpp - Cherry dialect ops -------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/CherryOps.h"
#include "cherry/MLIRGen/CherryDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace cherry {
#define GET_OP_CLASSES
#include "cherry/MLIRGen/CherryOps.cpp.inc"

auto ConstantOp::build(mlir::OpBuilder &builder,
                       mlir::OperationState &state,
                       uint64_t value) -> void {
  auto dataType = builder.getI64Type();
  auto dataAttribute = IntegerAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

auto CallOp::build(mlir::OpBuilder &builder,
                   mlir::OperationState &state,
                   StringRef callee,
                   ArrayRef<mlir::Value> arguments) -> void {
  auto dataType = builder.getI64Type();
  state.addTypes(dataType);
  state.addOperands(arguments);
  state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

auto PrintOp::build(mlir::OpBuilder &builder,
                   mlir::OperationState &state,
                   mlir::Value argument) -> void {
  auto dataType = builder.getI64Type();
  state.addTypes(dataType);
  state.addOperands({argument});
}

} // namespace cherry
} // namespace mlir
