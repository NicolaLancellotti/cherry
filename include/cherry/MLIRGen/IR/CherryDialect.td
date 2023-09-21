//===--- CherryDialect.td - Cherry dialect -----------------*- tablegen -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_DIALECT
#define CHERRY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Cherry dialect definition.
//===----------------------------------------------------------------------===//

def Cherry_Dialect : Dialect {
  let name = "cherry";
  let summary = "Cherry MLIR dialect.";
  let cppNamespace = "::mlir::cherry";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base cherry operation definition.
//===----------------------------------------------------------------------===//

class Cherry_Op<string mnemonic, list<Trait> traits = []> :
        Op<Cherry_Dialect, mnemonic, traits>;

#endif // CHERRY_DIALECT
