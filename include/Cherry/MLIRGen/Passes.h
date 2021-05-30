//===--- Passes.h - MLIR Passes ---------------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_PASSES_H
#define CHERRY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace cherry {

auto createLowerToSCFPass() -> std::unique_ptr<mlir::Pass>;
auto createLowerToSCFAndStandardPass() -> std::unique_ptr<mlir::Pass>;
auto createLowerToLLVMPass() -> std::unique_ptr<mlir::Pass>;

} // end namespace cherry
} // end namespace mlir

#endif // CHERRY_PASSES_H
