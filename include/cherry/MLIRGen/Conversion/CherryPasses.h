//===--- CherryPasses.h - Cherry passes -------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_PASSES_H
#define CHERRY_PASSES_H

#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "cherry/MLIRGen/IR/CherryOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace cherry {
#define GEN_PASS_DECL
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"
} // namespace cherry
} // namespace mlir

#endif // CHERRY_PASSES_H
