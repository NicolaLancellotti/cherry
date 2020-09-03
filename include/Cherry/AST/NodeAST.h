#ifndef CHERRY_NODEAST_H
#define CHERRY_NODEAST_H

#include "llvm/Support/SMLoc.h"
#include <list>

namespace cherry {

class NodeAST {
public:
  explicit NodeAST(llvm::SMLoc location) : _location{location} {};

  virtual ~NodeAST() = default;

  auto location() const -> const llvm::SMLoc& {
    return _location;
  }

private:
  llvm::SMLoc _location;
};

}

#endif