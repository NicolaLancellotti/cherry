//===--- Passes.h - MLIR Passes ---------------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_PASSES_H
#define CHERRY_PASSES_H

namespace mlir {
class Pass;

namespace cherry {

auto createLowerToStandardPass() -> std::unique_ptr<mlir::Pass>;

} // end namespace toy
} // end namespace mlir


#endif // CHERRY_PASSES_H
