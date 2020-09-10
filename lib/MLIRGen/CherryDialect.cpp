//===- CherryDialect.cpp - Cherry dialect ---------------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/CherryDialect.h"
#include "cherry/MLIRGen/CherryOps.h"

using namespace mlir;
using namespace mlir::cherry;

CherryDialect::CherryDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "cherry/MLIRGen/CherryOps.cpp.inc"
      >();
}
