#include "cherry/Parse/Token.h"

using namespace cherry;

auto Token::getUInt64IntegerValue() const -> llvm::Optional<uint64_t> {
  uint64_t result = 0;
  if (spelling.getAsInteger(10, result))
    return llvm::None;
  return result;
}