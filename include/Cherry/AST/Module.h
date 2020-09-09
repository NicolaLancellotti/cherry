//===--- Module.h - Cherry Language Module AST ------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_MODULE_H
#define CHERRY_MODULE_H

#include "Node.h"

namespace cherry {
class Decl;

class Module : public Node {
public:
  explicit Module(llvm::SMLoc location,
                  VectorUniquePtr<Decl> declarations)
      : Node{location},
        _declarations{std::move(declarations)} {};

  auto declarations() const -> const VectorUniquePtr<Decl>& {
    return _declarations;
  }

private:
  VectorUniquePtr<Decl> _declarations;

public:
  auto begin() const -> decltype(_declarations.begin()) { return _declarations.begin(); }
  auto end() const -> decltype(_declarations.end()) { return _declarations.end(); }
};

} // end namespace cherry

#endif // CHERRY_MODULE_H
