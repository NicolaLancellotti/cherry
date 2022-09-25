//===--- Passes.h - Cherry passes -------------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_PASSES_H
#define CHERRY_PASSES_H

#include "cherry/MLIRGen/Conversion/ConvertCherryToArithCfFunc.h"
#include "cherry/MLIRGen/Conversion/ConvertCherryToLLVM.h"
#include "cherry/MLIRGen/Conversion/ConvertCherryToSCF.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "cherry/MLIRGen/Conversion/Passes.h.inc"

} // namespace mlir

#endif // CHERRY_PASSES_H
