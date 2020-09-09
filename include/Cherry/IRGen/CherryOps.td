//===--- CherryOps.h - Cherry dialect ops -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_OPS
#define CHERRY_OPS

include "CherryDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : Cherry_Op<"constant", [NoSideEffect]> {
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:
    ```mlir
    %0 = "cherry.constant"() {value = 1 : i32} : () -> i32
    ```
  }];

  let arguments = (ins I32Attr:$value);
  let results = (outs I32);

  let builders = [
   OpBuilder<"OpBuilder &builder, OperationState &state, uint64_t value">
  ];
}

def ReturnOp : Cherry_Op<"return", [NoSideEffect, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation produces no results.
    For example:
    ```mlir
    func @bar() {
      "cherry.return"() : () -> ()
    }
    ```
  }];
}

def CallOp : Cherry_Op<"call"> {
  let summary = "call operation";
  let description = [{
    Calls represent calls to a user defined function. The callee name is
    attached as a symbol reference via an attribute. The arguments list must
    match the arguments expected by the callee. For example:
    ```mlir
    %3 = "cherry.call"(%1, %2) {callee = @bar} : (i32, i32) -> none
    ```
    This is only valid if a function named "bar" exists and takes two
    arguments.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,  Variadic<I32>:$inputs);
  let results = (outs NoneType);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "StringRef callee, ArrayRef<Value> arguments">
  ];
}

#endif // CHERRY_OPS
