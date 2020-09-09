//===--- CherryOps.h - Cherry dialect ops -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_CHERRYOPS_H
#define CHERRY_CHERRYOPS_H

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace cherry {

#define GET_OP_CLASSES
#include "cherry/IRGen/CherryOps.h.inc"

} // namespace cherry
} // namespace mlir

#endif // CHERRY_CHERRYOPS_H
