//===- CherryDialect.cpp - Cherry dialect ---------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/IRGen/CherryDialect.h"
#include "cherry/IRGen/CherryOps.h"

using namespace mlir;
using namespace mlir::cherry;

CherryDialect::CherryDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "cherry/IRGen/CherryOps.cpp.inc"
      >();
}
