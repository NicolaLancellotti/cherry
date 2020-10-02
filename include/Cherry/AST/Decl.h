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
class VariableDeclExpr;

// _____________________________________________________________________________
// Declaration

class Decl : public Node {
public:
  enum DeclarationKind {
    Decl_Function,
    Decl_Struct,
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
// Function declaration

class Prototype final : public Node {
public:
  explicit Prototype(llvm::SMLoc location,
                     std::unique_ptr<Identifier> id,
                     VectorUniquePtr<VariableDeclExpr> parameters,
                     std::unique_ptr<Identifier> type)
      : Node{location},
        _id(std::move(id)),
        _parameters{std::move(parameters)},
        _type{std::move(type)} {};

  auto id() const -> const std::unique_ptr<Identifier>& {
    return _id;
  }

  auto parameters() const -> const VectorUniquePtr<VariableDeclExpr>& {
    return _parameters;
  }

  auto type() const -> const std::unique_ptr<Identifier>& {
    return _type;
  }

private:
  std::unique_ptr<Identifier> _id;
  VectorUniquePtr<VariableDeclExpr> _parameters;
  std::unique_ptr<Identifier> _type;
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
                      VectorUniquePtr<VariableDeclExpr> variables)
      : Decl{Decl_Struct, location}, _id(std::move(id)),
        _variables(std::move(variables)) {};

  static auto classof(const Decl *node) -> bool {
    return node->getKind() == Decl_Struct;
  }

  auto id() const -> const std::unique_ptr<Identifier>& {
    return _id;
  }

  auto variables() const -> const VectorUniquePtr<VariableDeclExpr>& {
    return _variables;
  }

private:
  std::unique_ptr<Identifier> _id;
  VectorUniquePtr<VariableDeclExpr> _variables;

public:
  auto begin() const -> decltype(_variables.begin()) { return _variables.begin(); }
  auto end() const -> decltype(_variables.end()) { return _variables.end(); }
};

} // end namespace cherry

#endif // CHERRY_DECL_H
