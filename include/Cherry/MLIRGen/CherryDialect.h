//===--- CherryDialect.h - Cherry dialect -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_CHERRYDIALECT_H
#define CHERRY_CHERRYDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace cherry {

#include "cherry/MLIRGen/CherryOpsDialect.h.inc"

} // end namespace cherry
} // end namespace mlir

#endif // CHERRY_CHERRYDIALECT_H
