//===--- ASTDumper.cpp - Cherry Language AST Dumper ------------------------===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace cherry;

namespace {

using llvm::errs;

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

#define INDENT()                                                               \
  Indent level_(_curIndent);                                                    \
  indent();

class Dumper {
public:
  Dumper(const llvm::SourceMgr &sourceManager): _sourceManager{sourceManager} {}

  auto dump(const Module *node) -> void;

private:
  auto indent() -> void {
    for (int i = 0; i < _curIndent; i++)
      errs() << "  ";
  }

  int _curIndent = 0;
  const llvm::SourceMgr &_sourceManager;

  template <typename T>
  auto loc(T *node) -> std::string {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return (llvm::Twine("loc=") + llvm::Twine(line) + llvm::Twine(":")
            + llvm::Twine(col)).str();
  }

  // Declarations
  auto dump(const Decl *node) -> void;
  auto dump(const VariableDecl *node) -> void;
  auto dump(const Prototype *node) -> void;
  auto dump(const FunctionDecl *node) -> void;
  auto dump(const StructDecl *node) -> void;

  // Expressions
  auto dump(const Expr *node) -> void;
  auto dump(const CallExpr *node) -> void;
  auto dump(const DecimalExpr *node) -> void;
  auto dump(const StructExpr *node) -> void;
  auto dump(const VariableExpr *node) -> void;
  auto dump(const BinaryExpr *node) -> void;
};

}// end namespace

auto Dumper::dump(const Module *node) -> void {
  for (auto &decl : *node)
    dump(decl.get());
}

// Declarations

auto Dumper::dump(const Decl *node) -> void {
  llvm::TypeSwitch<const Decl *>(node)
      .Case<FunctionDecl, StructDecl>([&](auto *node) {
        this->dump(node);
      })
      .Default([&](const Decl *) {
        llvm_unreachable("Unexpected declaration");
      });
}

auto Dumper::dump(const VariableDecl *node) -> void {
  auto id = node->variable().get();
  auto type = node->type().get();
  INDENT();
  errs() << "Variable (id=" << id->name() << " " << loc(id)
         << ") (type=" << type->name() << " " << loc(type) << ")\n";
}

// Functions

auto Dumper::dump(const Prototype *node) -> void {
  auto id = node->id().get();
  INDENT();
  errs() << "Prototype " << loc(node)
         << " (name="<< id->name() << " " << loc(id) << ")\n";
  for (auto& parameter : node->parameters())
    dump(parameter.get());
}

auto Dumper::dump(const FunctionDecl *node) -> void {
  INDENT();
  errs() << "FunctionDecl " << loc(node) << "\n";
  dump(node->proto().get());
  for (auto &expr : *node)
    dump(expr.get());
}

auto Dumper::dump(const StructDecl *node) -> void {
  INDENT();
  auto id = node->id().get();
  errs() << "StructDecl " << loc(node)
         << " (name="<< id->name() << " " << loc(id) << ")\n";
  for (auto& var : node->variables())
    dump(var.get());
}

// Expressions

auto Dumper::dump(const Expr *node) -> void {
  llvm::TypeSwitch<const Expr *>(node)
      .Case<CallExpr, DecimalExpr, VariableExpr,
            StructExpr, BinaryExpr>([&](auto *node) {
        this->dump(node);
      })
      .Default([&](const Expr *) {
        llvm_unreachable("Unexpected expression");
      });
}

auto Dumper::dump(const CallExpr *node) -> void {
  INDENT();
  errs() << "CallExpr " << loc(node) << " callee=" << node->name() << "\n";
  for (auto &expr : *node)
    dump(expr.get());
}

auto Dumper::dump(const DecimalExpr *node) -> void {
  INDENT();
  errs() << "DecimalExpr " << loc(node) << " value=" << node->value() << "\n";
}

auto Dumper::dump(const StructExpr *node) -> void {
  INDENT();
  errs() << "StructExpr " << loc(node) << " type=" << node->type() << "\n";
  for (auto &expr : *node)
    dump(expr.get());
}

auto Dumper::dump(const VariableExpr *node) -> void {
  INDENT();
  errs() << "VariableExpr " << loc(node) << " name=" << node->name() << "\n";
}

auto Dumper::dump(const BinaryExpr *node) -> void {
  INDENT();
  errs() << "BinaryExpr " << loc(node) << " op=" << node->op() << "\n";
  dump(node->lhs().get());
  dump(node->rhs().get());
}

namespace cherry {

auto dumpAST(const llvm::SourceMgr &sourceManager,
             const Module &module) -> void {
  Dumper(sourceManager).dump(&module);
}

} // end namespace cherry
