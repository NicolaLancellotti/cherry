#ifndef CHERRY_AST_H
#define CHERRY_AST_H

#include "cherry/AST/NodeAST.h"
#include "cherry/AST/DeclarationsAST.h"

namespace cherry {

class ModuleAST : public NodeAST {
public:
  explicit ModuleAST(Location location,
                     std::unique_ptr<DeclarationAST> declarations)
      : NodeAST{location},
        _declarations{std::move(declarations)} {};

  auto declarations() const -> const std::unique_ptr<DeclarationAST>& {
    return _declarations;
  }

private:
  std::unique_ptr<DeclarationAST> _declarations;
};

}

#endif