#include "cherry/Parse/Token.h"

using namespace cherry;

llvm::Optional<uint64_t> Token::getUInt64IntegerValue(llvm::StringRef spelling) {
  uint64_t result = 0;
  if (spelling.getAsInteger(10, result))
    return llvm::None;
  return result;
}