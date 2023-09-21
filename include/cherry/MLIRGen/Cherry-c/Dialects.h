//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_C_DIALECTS_H
#define CHERRY_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Cherry, cherry);

#ifdef __cplusplus
}
#endif

#endif // CHERRY_C_DIALECTS_H
