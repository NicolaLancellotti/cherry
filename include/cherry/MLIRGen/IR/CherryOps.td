//===--- CherryOps.h - Cherry dialect ops -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_OPS
#define CHERRY_OPS

include "CherryDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Cherry_Type : AnyTypeOf<[I64, I1]>;

def ConstantOp : Cherry_Op<"constant", [NoSideEffect]> {
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:
    ```mlir
    %0 = "cherry.constant"() {value = 1 : i64} : () -> iI64
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs Cherry_Type);

  let builders = [
    OpBuilder<(ins "uint64_t":$value)>,
    OpBuilder<(ins "bool":$value)>
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
      ...
      "cherry.return"(%1) : (i64) -> ()
    }
    ```
  }];

  let arguments = (ins Variadic<Cherry_Type>:$input);
}

def CallOp : Cherry_Op<"call"> {
  let summary = "call operation";
  let description = [{
    Calls represent calls to a user defined function. The callee name is
    attached as a symbol reference via an attribute. The arguments list must
    match the arguments expected by the callee. For example:
    ```mlir
    %1 = "cherry.call"() {callee = @bar} : () -> i64
    ```
    This is only valid if a function named "bar" exists and takes two
    arguments.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,  Variadic<Cherry_Type>:$inputs);
  let results = (outs Variadic<Cherry_Type>);

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments,
                      "ArrayRef<Type>":$results)>
  ];
}

def PrintOp : Cherry_Op<"print"> {
  let summary = "print operation";
  let description = [{
      The "print" builtin operation prints a given input
  }];

  let arguments = (ins AnyTypeOf<[I64, I64MemRef]>:$input);
  let results = (outs I64);
}

def CastOp : Cherry_Op<"cast"> {
  let summary = "cast operation";
  let description = [{
    Casts from I64 to I1
  }];

  let arguments = (ins AnyTypeOf<[I1, I1MemRef]>:$input);
  let results = (outs I64);
}

def IfOp : Cherry_Op<"if"> {
  let summary = "if-then-else operation";

  let arguments = (ins I1:$condition);
  let results = (outs Cherry_Type);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$cond,
                    "function_ref<void(OpBuilder &, Location)>":$thenBuilder,
                    "function_ref<void(OpBuilder &, Location)>":$elseBuilder)>
  ];
}

def YieldIfOp : Cherry_Op<"yield_if", [NoSideEffect, Terminator]> {
  let summary = "loop yield and termination operation";

  let arguments = (ins Variadic<Cherry_Type>:$result);

  let builders = [
    OpBuilder<(ins ), [{ /* nothing to do */ }]>
  ];
}

def WhileOp : Cherry_Op<"while"> {
  let summary = "while operation";

  let regions = (region SizedRegion<1>:$conditionRegion, SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "function_ref<void(OpBuilder &, Location)>":$conditionBuilder,
                    "function_ref<void(OpBuilder &, Location)>":$bodyBuilder)>
  ];
}

def YieldWhileOp : Cherry_Op<"yield_while", [NoSideEffect, Terminator]> {
  let summary = "loop yield and termination operation";

  let arguments = (ins Variadic<Cherry_Type>:$result);

  let builders = [
    OpBuilder<(ins ), [{ /* nothing to do */ }]>
  ];
}

def ArithmeticLogicOp : Cherry_Op<"arithmetic-logic", [NoSideEffect]> {

  let arguments = (ins Cherry_Type:$lhs, Cherry_Type:$rhs, StrAttr:$op);
  let results = (outs Cherry_Type);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "StringRef":$op, "Type":$type)>
  ];
}

#endif // CHERRY_OPS
