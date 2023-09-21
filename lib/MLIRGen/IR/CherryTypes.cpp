//===- CherryTypes.cpp - Cherry dialect types -------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/CherryTypes.h"
#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::cherry;

#define GET_TYPEDEF_CLASSES
#include "cherry/MLIRGen/IR/CherryOpsTypes.cpp.inc"

void CherryDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cherry/MLIRGen/IR/CherryOpsTypes.cpp.inc"
      >();
}
