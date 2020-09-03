#ifndef CHERRY_DECLARATIONSAST_H
#define CHERRY_DECLARATIONSAST_H

#include "cherry/AST/NodeAST.h"
#include <string>

namespace cherry {

// _____________________________________________________________________________
// DeclarationAST

class DeclarationAST: public NodeAST {
public:
  explicit DeclarationAST(llvm::SMLoc location): NodeAST{location} {};
};

// _____________________________________________________________________________
// FunctionDeclAST

class PrototypeAST : public NodeAST {
public:
  explicit PrototypeAST(llvm::SMLoc location,
                        std::string name)
      : NodeAST{location}, _name(std::move(name)) {};

  auto name() const -> const std::string& {
    return _name;
  }

private:
  std::string _name;
};

class FunctionDeclAST: public DeclarationAST {
public:
  explicit FunctionDeclAST(llvm::SMLoc location,
                           std::unique_ptr<PrototypeAST> proto,
                           std::unique_ptr<DeclarationAST> body)
      : DeclarationAST{location}, _proto(std::move(proto)),
        _body(std::move(body)) {};

  auto proto() const -> const std::unique_ptr<PrototypeAST>& {
    return _proto;
  }

  auto body() const -> const std::unique_ptr<DeclarationAST>& {
    return _body;
  }

private:
  std::unique_ptr<PrototypeAST> _proto;
  std::unique_ptr<DeclarationAST> _body;
};

}

#endif