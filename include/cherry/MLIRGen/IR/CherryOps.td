//===--- CherryOps.h - Cherry dialect ops -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_OPS
#define CHERRY_OPS

include "cherry/MLIRGen/IR/CherryTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Cherry_Type : AnyTypeOf<[I64, I1, Cherry_Struct]>;

def ConstantOp : Cherry_Op<"constant", [Pure]> {
  let summary = "Constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:
    ```mlir
    %0 = "cherry.constant"() {value = 1 : i64} : () -> iI64
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs Cherry_Type);

  let builders = [
    OpBuilder<(ins "uint64_t":$value)>,
    OpBuilder<(ins "bool":$value)>
  ];

  let assemblyFormat = "$value attr-dict `:` type(results)";
}

def ReturnOp : Cherry_Op<"return", [Pure, Terminator]> {
  let summary = "Return operation";
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

  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
}

def CallOp : Cherry_Op<"call"> {
  let summary = "Call operation";
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

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def PrintOp : Cherry_Op<"print"> {
  let summary = "Print operation";
  let description = [{
      The "print" builtin operation prints a given input
  }];

  let arguments = (ins AnyTypeOf<[I64, I64MemRef]>:$input);
  let results = (outs I64:$result);

  let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";
}

def CastOp : Cherry_Op<"cast"> {
  let summary = "Cast operation";
  let description = [{
    Casts from I64 to I1
  }];

  let arguments = (ins AnyTypeOf<[I1, I1MemRef]>:$input);
  let results = (outs I64:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

def IfOp : Cherry_Op<"if"> {
  let summary = "If-then-else operation";

  let arguments = (ins I1:$condition);
  let results = (outs Cherry_Type:$result);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$cond,
                    "function_ref<void(OpBuilder &, Location)>":$thenBuilder,
                    "function_ref<void(OpBuilder &, Location)>":$elseBuilder)>
  ];

  let assemblyFormat = [{
    $condition `:` type($result) $thenRegion `else` $elseRegion attr-dict 
  }];
}

def YieldIfOp : Cherry_Op<"yield_if", [Pure, Terminator]> {
  let summary = "Yield if operation";

  let arguments = (ins Variadic<Cherry_Type>:$result);

  let builders = [
    OpBuilder<(ins ), [{ /* nothing to do */ }]>
  ];

  let assemblyFormat = "($result^ `:` type($result))? attr-dict";
}

def WhileOp : Cherry_Op<"while"> {
  let summary = "While operation";

  let regions = (region SizedRegion<1>:$conditionRegion, SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "function_ref<void(OpBuilder &, Location)>":$conditionBuilder,
                    "function_ref<void(OpBuilder &, Location)>":$bodyBuilder)>
  ];

  let assemblyFormat = "$conditionRegion `do` $bodyRegion attr-dict";
}

def YieldWhileOp : Cherry_Op<"yield_while", [Pure, Terminator]> {
  let summary = "Yield while operation";

  let arguments = (ins Variadic<Cherry_Type>:$result);

  let builders = [
    OpBuilder<(ins ), [{ /* nothing to do */ }]>
  ];

  let assemblyFormat = "($result^ `:` type($result))? attr-dict";
}

def ArithmeticLogicOp : Cherry_Op<"arithmetic-logic", [Pure]> {
  let summary = "Arithmetic logic operation";

  let arguments = (ins Cherry_Type:$lhs, Cherry_Type:$rhs, StrAttr:$op);
  let results = (outs Cherry_Type:$result);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "StringRef":$op, "Type":$type)>
  ];

  let assemblyFormat = [{
    $rhs $op $lhs attr-dict `:` `(` type($rhs) `,` type($lhs) `)` `->` type($result)
  }];
}

def StructInitOp : Cherry_Op<"struct_init"> {
  let summary = "Struct initializer";

  let arguments = (ins Variadic<Cherry_Type>:$values);
  let results = (outs Cherry_Struct:$result);

  let assemblyFormat = [{
    `(` $values `)` attr-dict `:` functional-type($values, $result)
  }];
}

def StructReadOp : Cherry_Op<"struct_read", [Pure]> {
  let summary = "Struct read";
  
  let arguments = (ins Cherry_Struct:$structValue, I64Attr:$index);
  let results = (outs Cherry_Type:$result);

  let builders = [
    OpBuilder<(ins "Value":$structValue, "int64_t":$index)>
  ];

  let assemblyFormat = [{
    $structValue `[` $index `]` attr-dict `:` functional-type($structValue, $result)
  }];
}

def StructWriteOp : Cherry_Op<"struct_write", [Pure]> {
  let summary = "Struct write";
  
  let arguments = (ins Cherry_Struct:$structValue, DenseI64ArrayAttr:$indexes, Cherry_Type:$valueToStore);
  let results = (outs Cherry_Struct:$result);

  let builders = [
     OpBuilder<(ins "Value":$structValue, "ArrayRef<int64_t>":$indexes, "Value":$valueToStore)>
  ];

  let assemblyFormat = [{
     $valueToStore `,` $structValue $indexes attr-dict `:` functional-type(operands, $result)
  }];
}

#endif // CHERRY_OPS
