//===- CherryDialect.cpp - Cherry dialect ---------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "cherry/MLIRGen/IR/CherryOps.h"
#include "cherry/MLIRGen/IR/CherryTypes.h"

using namespace mlir;
using namespace mlir::cherry;

#include "cherry/MLIRGen/IR/CherryOpsDialect.cpp.inc"

void CherryDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cherry/MLIRGen/IR/CherryOps.cpp.inc"
      >();
  registerTypes();
}
