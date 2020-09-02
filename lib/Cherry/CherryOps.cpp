//===- CherryOps.cpp - Cherry dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Cherry/CherryOps.h"
#include "Cherry/CherryDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace cherry {
#define GET_OP_CLASSES
#include "Cherry/CherryOps.cpp.inc"
} // namespace cherry
} // namespace mlir
