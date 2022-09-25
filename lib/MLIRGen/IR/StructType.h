//===--- StructType.h - MLIR lowering of Cherry struct type -----*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_STRUCTTYPE_H
#define CHERRY_STRUCTTYPE_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace cherry {
namespace detail {

struct StructTypeStorage : public mlir::TypeStorage {

  using KeyTy = llvm::ArrayRef<mlir::Type>;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  llvm::ArrayRef<mlir::Type> elementTypes;
};

} // end namespace detail

class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
  using Base::Base;

  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  llvm::ArrayRef<mlir::Type> getElementTypes() {
    return getImpl()->elementTypes;
  }

  size_t getNumElementTypes() { return getElementTypes().size(); }
};

} // end namespace cherry
} // end namespace mlir

#endif // CHERRY_STRUCTTYPE_H
