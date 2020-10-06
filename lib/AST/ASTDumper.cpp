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

class Dumper {
public:
  Dumper(const llvm::SourceMgr &sourceManager): _sourceManager{sourceManager} {}

  auto dump(const Module *node) -> void;

private:
  int _curIndent = 0;
  const llvm::SourceMgr &_sourceManager;

  // Declarations
  auto dump(const Decl *node) -> void;
  auto dump(const Prototype *node) -> void;
  auto dump(const FunctionDecl *node) -> void;
  auto dump(const StructDecl *node) -> void;

  // Expressions
  auto dump(const Expr *node) -> void;
  auto dump(const VectorUniquePtr<Expr> &node, llvm::StringRef string) -> void;
  auto dump(const CallExpr *node) -> void;
  auto dump(const VariableDeclExpr *node) -> void;
  auto dump(const VariableExpr *node) -> void;
  auto dump(const DecimalLiteralExpr *node) -> void;
  auto dump(const BoolLiteralExpr *node) -> void;
  auto dump(const BinaryExpr *node) -> void;
  auto dump(const IfExpr *node) -> void;

  // Utility
  auto indent() -> void {
    for (int i = 0; i < _curIndent; i++)
      errs() << "  ";
  }

  template <typename T>
  auto loc(T *node) -> std::string {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return (llvm::Twine("loc=") + llvm::Twine(line) + llvm::Twine(":")
            + llvm::Twine(col)).str();
  }
};

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

#define INDENT()                                                               \
  Indent level_(_curIndent);                                                    \
  indent();

}// end namespace

auto Dumper::dump(const Module *node) -> void {
  for (auto &decl : *node)
    dump(decl.get());
}

auto Dumper::dump(const Decl *node) -> void {
  llvm::TypeSwitch<const Decl *>(node)
      .Case<FunctionDecl, StructDecl>([&](auto *node) {
        this->dump(node);
      })
      .Default([&](const Decl *) {
        llvm_unreachable("Unexpected declaration");
      });
}

auto Dumper::dump(const Prototype *node) -> void {
  auto id = node->id().get();
  auto type = node->type().get();
  INDENT();
  errs() << "Prototype " << loc(node)
         << " (name="<< id->name() << " " << loc(id)
         << " (type="<< type->name() << " " << loc(type) << ")\n";
  for (auto &parameter : node->parameters())
    dump(parameter.get());
}

auto Dumper::dump(const FunctionDecl *node) -> void {
  INDENT();
  errs() << "FunctionDecl " << loc(node) << "\n";
  dump(node->proto().get());
  dump(node->body(), "Body:");
}

auto Dumper::dump(const StructDecl *node) -> void {
  INDENT();
  auto id = node->id().get();
  errs() << "StructDecl " << loc(node)
         << " (name="<< id->name() << " " << loc(id) << ")\n";
  for (auto &var : node->variables())
    dump(var.get());
}

auto Dumper::dump(const Expr *node) -> void {
  llvm::TypeSwitch<const Expr *>(node)
      .Case<CallExpr, DecimalLiteralExpr, BoolLiteralExpr, VariableExpr, IfExpr,
          BinaryExpr, VariableDeclExpr>([&](auto *node) {
        this->dump(node);
      })
      .Default([&](const Expr *) {
        llvm_unreachable("Unexpected expression");
      });
}

auto Dumper::dump(const VectorUniquePtr<Expr> &node,
                  llvm::StringRef string) -> void {
  INDENT();
  errs() << string << "\n";
  for (auto &expr : node)
    dump(expr.get());
}

auto Dumper::dump(const CallExpr *node) -> void {
  INDENT();
  errs() << "CallExpr " << loc(node) << " type=" << node->type()
         << " callee=" << node->name() << "\n";
  for (auto &expr : *node)
    dump(expr.get());
}

auto Dumper::dump(const VariableDeclExpr *node) -> void {
  auto id = node->variable().get();
  auto varType = node->varType().get();
  INDENT();
  errs() << "VariableDeclExpr (id=" << id->name() << " " << loc(id)
         << ") (type=" << varType->name() << " " << loc(varType) << ")\n";
  if (node->init())
    dump(node->init().get());
}

auto Dumper::dump(const VariableExpr *node) -> void {
  INDENT();
  errs() << "VariableExpr " << loc(node) << " type=" << node->type()
         << " name=" << node->name() << "\n";
}

auto Dumper::dump(const DecimalLiteralExpr *node) -> void {
  INDENT();
  errs() << "DecimalExpr " << loc(node) << " type=" << node->type()
         << " value=" << node->value() << "\n";
}

auto Dumper::dump(const BoolLiteralExpr *node) -> void {
  INDENT();
  errs() << "BoolLiteralExpr " << loc(node) << " type=" << node->type()
         << " value=" << node->value() << "\n";
}

auto Dumper::dump(const BinaryExpr *node) -> void {
  INDENT();
  errs() << "BinaryExpr " << loc(node) << " type=" << node->type()
         <<" op=`" << node->op() << "`\n";
  dump(node->lhs().get());
  dump(node->rhs().get());
}

auto Dumper::dump(const IfExpr *node) -> void {
  INDENT();
  errs() << "IfExpr " << loc(node) << " type=" << node->type() << "`\n";
  dump(node->conditionExpr().get());
  dump(node->thenExpr(), "thenExpr:");
  dump(node->elseExpr(), "elseExpr:");
}

namespace cherry {

auto dumpAST(const llvm::SourceMgr &sourceManager,
             const Module &module) -> void {
  Dumper(sourceManager).dump(&module);
}

} // end namespace cherry
