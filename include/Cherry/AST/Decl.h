//===--- Decl.h - Cherry Language Declaration ASTs --------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_DECL_H
#define CHERRY_DECL_H

#include "Node.h"
#include "llvm/ADT/StringRef.h"

namespace cherry {

class Expr;
class VariableExpr;

// _____________________________________________________________________________
// Declaration

class Decl : public Node {
public:
  enum DeclarationKind {
    Decl_Function,
    Decl_Struct,
    Decl_Var,
  };

  explicit Decl(DeclarationKind kind,
                llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> DeclarationKind { return _kind; }

private:
  const DeclarationKind _kind;
};

// _____________________________________________________________________________
// Identifier

class Identifier final : public Node {
public:
  explicit Identifier(llvm::SMLoc location,
                      llvm::StringRef name)
      : Node{location}, _name(name.str()) {};

  auto name() const -> llvm::StringRef {
    return _name;
  }

private:
  std::string _name;
};

// _____________________________________________________________________________
// VariableDecl

class VariableDecl final : public Decl {
public:
  explicit VariableDecl(llvm::SMLoc location,
                        std::unique_ptr<VariableExpr> variable,
                        std::unique_ptr<Identifier> type)
      : Decl{Decl_Var, location}, _variable(std::move(variable)),
        _type(std::move(type)) {};

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Var;
  }

  auto variable() const -> const std::unique_ptr<VariableExpr>& {
    return _variable;
  }

  auto type() const -> const std::unique_ptr<Identifier>& {
    return _type;
  }

private:
  std::unique_ptr<VariableExpr> _variable;
  std::unique_ptr<Identifier> _type;
};

// _____________________________________________________________________________
// Function declaration

class Prototype final : public Node {
public:
  explicit Prototype(llvm::SMLoc location,
                     std::unique_ptr<Identifier> id,
                     VectorUniquePtr<VariableDecl> parameters)
      : Node{location},
        _id(std::move(id)),
        _parameters{std::move(parameters)} {};

  auto id() const -> const std::unique_ptr<Identifier>& {
    return _id;
  }

  auto parameters() const -> const VectorUniquePtr<VariableDecl>& {
    return _parameters;
  }

private:
  std::unique_ptr<Identifier> _id;
  VectorUniquePtr<VariableDecl> _parameters;
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

// _____________________________________________________________________________
// Struct declaration

class StructDecl final : public Decl {
public:
  explicit StructDecl(llvm::SMLoc location,
                      std::unique_ptr<Identifier> id,
                      VectorUniquePtr<VariableDecl> variables)
      : Decl{Decl_Struct, location}, _id(std::move(id)),
        _variables(std::move(variables)) {};

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Struct;
  }

  auto id() const -> const std::unique_ptr<Identifier>& {
    return _id;
  }

  auto variables() const -> const VectorUniquePtr<VariableDecl>& {
    return _variables;
  }

private:
  std::unique_ptr<Identifier> _id;
  VectorUniquePtr<VariableDecl> _variables;

public:
  auto begin() const -> decltype(_variables.begin()) { return _variables.begin(); }
  auto end() const -> decltype(_variables.end()) { return _variables.end(); }
};

} // end namespace cherry

#endif // CHERRY_DECL_H
