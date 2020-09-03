#ifndef CHERRY_MODULEAST_H
#define CHERRY_MODULEAST_H

#include "cherry/AST/NodeAST.h"
#include "cherry/AST/DeclarationsAST.h"

namespace cherry {

class ModuleAST : public NodeAST {
public:
  explicit ModuleAST(llvm::SMLoc location,
                     std::vector<DeclarationAST> declarations)
      : NodeAST{location},
        _declarations{std::move(declarations)} {};

  auto declarations() const -> const std::vector<DeclarationAST>& {
    return _declarations;
  }

private:
  std::vector<DeclarationAST> _declarations;
};

}

#endif