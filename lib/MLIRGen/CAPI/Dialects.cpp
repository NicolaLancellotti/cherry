//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Cherry-c/Dialects.h"
#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Cherry, cherry,
                                      mlir::cherry::CherryDialect)
