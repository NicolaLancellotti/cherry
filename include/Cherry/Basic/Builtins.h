//===--- Builtins.h -------------------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_BUILTINS_H
#define CHERRY_BUILTINS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace cherry {
namespace builtins {

// Functions
const llvm::StringRef print = "print";
const llvm::StringRef boolToUInt64 = "boolToUInt64";

// Types
const llvm::StringRef UnitType = "Unit";
const llvm::StringRef BoolType = "Bool";
const llvm::StringRef UInt64Type = "UInt64";

inline auto primitiveTypes() -> llvm::SmallVector<llvm::StringRef, 2> {
  return {UnitType, BoolType, UInt64Type};
}

} // end namespace builtins
} // end namespace cherry

#endif // CHERRY_BUILTINS_H
