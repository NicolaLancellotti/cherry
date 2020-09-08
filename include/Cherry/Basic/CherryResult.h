#ifndef CHERRY_CHERRYRESULT_H
#define CHERRY_CHERRYRESULT_H

#include "mlir/Support/LogicalResult.h"

namespace cherry {

class CherryResult : public mlir::LogicalResult {
public:
  CherryResult(LogicalResult result = mlir::success()) : mlir::LogicalResult(result) {}

  explicit operator bool() const { return failed(*this); }
};

}
#endif
