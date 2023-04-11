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
  let cppNamespace = "::mlir::cherry";
  let useFoldAPI = kEmitFoldAdaptorFolder;

  let extraClassDeclaration = [{
    auto parseType(mlir::DialectAsmParser &parser) const -> mlir::Type override;
    auto printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const -> void override;
  }];
}

//===----------------------------------------------------------------------===//
// Base cherry operation definition.
//===----------------------------------------------------------------------===//

class Cherry_Op<string mnemonic, list<Trait> traits = []> :
        Op<Cherry_Dialect, mnemonic, traits>;

#endif // CHERRY_DIALECT
