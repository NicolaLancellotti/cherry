#ifndef CHERRY_MODULE_H
#define CHERRY_MODULE_H

#include "cherry/AST/Node.h"
#include "cherry/AST/Declarations.h"

namespace cherry {

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

}

#endif