//===--- Node.h - Cherry Language Node AST ----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_NODE_H
#define CHERRY_NODE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"

namespace cherry {

template<typename T> using VectorUniquePtr = llvm::SmallVector<std::unique_ptr<T>, 2>;

class Node {
public:
  explicit Node(llvm::SMLoc location) : _location{location} {};

  virtual ~Node() = default;

  auto location() const -> const llvm::SMLoc& {
    return _location;
  }

private:
  llvm::SMLoc _location;
};

}

#endif // CHERRY_NODE_H
