#ifndef CHERRY_OPS
#define CHERRY_OPS

include "CherryDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : Cherry_Op<"constant", [NoSideEffect]> {
  let arguments = (ins I32Attr:$value);
  let results = (outs I32);

  let builders = [
   OpBuilder<"OpBuilder &builder, OperationState &state, uint64_t value">
  ];
}

def ReturnOp : Cherry_Op<"return", [NoSideEffect, Terminator]> {

}

def CallOp : Cherry_Op<"call"> {

  let arguments = (ins FlatSymbolRefAttr:$callee,  Variadic<I32>:$inputs);
  let results = (outs NoneType);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "StringRef callee, ArrayRef<Value> arguments">
  ];

}
#endif
