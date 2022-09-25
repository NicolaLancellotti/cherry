//===--- ConvertCherryToArithCfFunc.h ---------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_CONVERTCHERRYTOARITHCFFUNC_H
#define CHERRY_CONVERTCHERRYTOARITHCFFUNC_H

#include <memory>

namespace mlir {
class Pass;

namespace cherry {
auto createConvertCherryToArithCfFunc() -> std::unique_ptr<mlir::Pass>;
} // end namespace cherry
} // end namespace mlir

#endif // CHERRY_CONVERTCHERRYTOARITHCFFUNC_H
