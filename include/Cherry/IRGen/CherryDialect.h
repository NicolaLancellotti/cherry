//===- CherryDialect.h - Cherry dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_CHERRYDIALECT_H
#define CHERRY_CHERRYDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace cherry {

#include "cherry/IRGen/CherryOpsDialect.h.inc"

} // namespace cherry
} // namespace mlir

#endif // CHERRY_CHERRYDIALECT_H
