

//===--- CherryResult.h - Cherry subclass of LogicalResult ------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_CHERRYRESULT_H
#define CHERRY_CHERRYRESULT_H

#include "llvm/Support/LogicalResult.h"

namespace cherry {

class CherryResult : public llvm::LogicalResult {
public:
  CherryResult(LogicalResult result = llvm::success())
      : llvm::LogicalResult(result) {}

  explicit operator bool() const { return llvm::failed(*this); }
};

} // end namespace cherry

#endif // CHERRY_CHERRYRESULT_H
