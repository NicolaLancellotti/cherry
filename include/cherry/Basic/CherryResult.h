

//===--- CherryResult.h - Cherry subclass of LogicalResult ------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_CHERRYRESULT_H
#define CHERRY_CHERRYRESULT_H

#include "mlir/Support/LogicalResult.h"

namespace cherry {

class CherryResult : public mlir::LogicalResult {
public:
  CherryResult(LogicalResult result = mlir::success())
      : mlir::LogicalResult(result) {}

  explicit operator bool() const { return mlir::failed(*this); }
};

} // end namespace cherry

#endif // CHERRY_CHERRYRESULT_H
