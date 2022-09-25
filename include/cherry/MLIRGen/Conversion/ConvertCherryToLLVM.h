//===--- ConvertCherryToLLVM.h ----------------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_CONVERTCHERRYTOLLVM_H
#define CHERRY_CONVERTCHERRYTOLLVM_H

#include <memory>

namespace mlir {
class Pass;

namespace cherry {
auto createConvertCherryToLLVMPass() -> std::unique_ptr<mlir::Pass>;
} // end namespace cherry
} // end namespace mlir

#endif // CHERRY_CONVERTCHERRYTOLLVM_H
