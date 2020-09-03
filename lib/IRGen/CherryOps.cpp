//===- CherryOps.cpp - Cherry dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cherry/IRGen/CherryOps.h"
#include "cherry/IRGen/CherryDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace cherry {
#define GET_OP_CLASSES
#include "cherry/IRGen/CherryOps.cpp.inc"
} // namespace cherry
} // namespace mlir
