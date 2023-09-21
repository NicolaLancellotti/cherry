//===- CherryTypes.td - Cherry dialect types ---------------*- tablegen -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_TYPES
#define CHERRY_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "cherry/MLIRGen/IR/CherryDialect.td"

//===----------------------------------------------------------------------===//
// Cherry type definitions
//===----------------------------------------------------------------------===//

class Cherry_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Cherry_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Cherry_Struct : Cherry_Type<"CherryStruct", "struct", [MemRefElementTypeInterface]> {
    let summary = "Cherry struct";
    let parameters = (ins ArrayRefParameter<"Type">:$types);
    let assemblyFormat = "`<` params `>`";
}

#endif // CHERRY_TYPES
