#ifndef CHERRY_PARSERESULT_H
#define CHERRY_PARSERESULT_H

#include "mlir/Support/LogicalResult.h"

namespace cherry {

class ParseResult : public mlir::LogicalResult {
public:
  ParseResult(LogicalResult result = mlir::success()) : LogicalResult(result) {}

  explicit operator bool() const { return failed(*this); }
};

}
#endif
