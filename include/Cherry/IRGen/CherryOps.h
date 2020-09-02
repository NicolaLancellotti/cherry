//===- CherryOps.h - Cherry dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_STANDALONEOPS_H
#define STANDALONE_STANDALONEOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace cherry {

#define GET_OP_CLASSES
#include "Cherry/CherryOps.h.inc"

} // namespace cherry
} // namespace mlir

#endif // STANDALONE_STANDALONEOPS_H
