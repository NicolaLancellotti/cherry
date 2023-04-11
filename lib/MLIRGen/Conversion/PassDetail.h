//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_PASSDETAIL_H
#define CHERRY_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // namespace arith

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace scf {
class SCFDialect;
} // namespace scf

#define GEN_PASS_CLASSES
#include "cherry/MLIRGen/Conversion/Passes.h.inc"

} // namespace mlir

#endif // CHERRY_PASSDETAIL_H
