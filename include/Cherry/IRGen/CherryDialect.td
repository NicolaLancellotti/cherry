//===- CherryDialect.td - Cherry dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    let cppNamespace = "cherry";
}

//===----------------------------------------------------------------------===//
// Base cherry operation definition.
//===----------------------------------------------------------------------===//

class Cherry_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Cherry_Dialect, mnemonic, traits>;

#endif // CHERRY_DIALECT
