//===--- Decl.h - Cherry Language Declaration ASTs --------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_DECL_H
#define CHERRY_DECL_H

#include "Node.h"
#include <string>
#include <vector>

namespace cherry {

class Expr;

// _____________________________________________________________________________
// Declaration

class Decl : public Node {
public:
  enum DeclarationKind {
    Decl_Function,
  };

  explicit Decl(DeclarationKind kind,
                       llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> DeclarationKind { return _kind; }

private:
  const DeclarationKind _kind;
};

// _____________________________________________________________________________
// Function declaration

class Identifier final : public Node {
public:
  explicit Identifier(llvm::SMLoc location,
                     std::string name)
      : Node{location}, _name(std::move(name)) {};

  auto name() const -> const std::string& {
    return _name;
  }

private:
  std::string _name;
};


using Parameter = std::pair<std::unique_ptr<Identifier>, std::unique_ptr<Identifier>>;

class Prototype final : public Node {
public:
  explicit Prototype(llvm::SMLoc location,
                     std::string name,
                     std::vector<Parameter> parameters)
      : Node{location}, _name(std::move(name)),
        _parameters{std::move(parameters)} {};

  auto name() const -> const std::string& {
    return _name;
  }

  auto parameters() const -> const std::vector<Parameter>& {
    return _parameters;
  }

private:
  std::string _name;
  std::vector<Parameter> _parameters;
};

class FunctionDecl final : public Decl {
public:
  explicit FunctionDecl(llvm::SMLoc location,
                        std::unique_ptr<Prototype> proto,
                        VectorUniquePtr<Expr> body)
      : Decl{Decl_Function, location}, _proto(std::move(proto)),
        _body(std::move(body)) {};

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Function;
  }

  auto proto() const -> const std::unique_ptr<Prototype>& {
    return _proto;
  }

  auto body() const -> const VectorUniquePtr<Expr>& {
    return _body;
  }

private:
  std::unique_ptr<Prototype> _proto;
  VectorUniquePtr<Expr> _body;

public:
  auto begin() const -> decltype(_body.begin()) { return _body.begin(); }
  auto end() const -> decltype(_body.end()) { return _body.end(); }
};

} // end namespace cherry

#endif // CHERRY_DECL_H
