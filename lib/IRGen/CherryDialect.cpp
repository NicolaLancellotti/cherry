//===- CherryDialect.cpp - Cherry dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cherry/IRGen/CherryDialect.h"
#include "cherry/IRGen/CherryOps.h"

using namespace mlir;
using namespace mlir::cherry;

//===----------------------------------------------------------------------===//
// Cherry dialect.
//===----------------------------------------------------------------------===//

CherryDialect::CherryDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "cherry/IRGen/CherryOps.cpp.inc"
      >();
}
