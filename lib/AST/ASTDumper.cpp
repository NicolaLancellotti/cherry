#include "cherry/AST/AST.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace cherry;

namespace {

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

class Dumper {
public:
  Dumper(const llvm::SourceMgr &sourceManager): _sourceManager{sourceManager} {}

  auto dump(const Module *node) -> void;

private:
  auto indent() -> void {
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
  }

  int curIndent = 0;
  const llvm::SourceMgr &_sourceManager;

  template <typename T>
  auto loc(T *node) -> std::string {
    auto [line, col] = _sourceManager.getLineAndColumn(node->location());
    return (llvm::Twine("loc=") + llvm::Twine(line) + llvm::Twine(":")
            + llvm::Twine(col)).str();
  }

  // Declarations
  auto dump(const Decl *node) -> void;
  auto dump(const Prototype *node) -> void;
  auto dump(const FunctionDecl *node) -> void;

  // Expressions
  auto dump(const Expr *node) -> void;
  auto dump(const CallExpr *node) -> void;
  auto dump(const DecimalExpr *node) -> void;
};

}

auto Dumper::dump(const Module *node) -> void {
  for (auto &decl : *node) {
    dump(decl.get());
  }
}

// Declarations

auto Dumper::dump(const Decl *node) -> void {
  llvm::TypeSwitch<const Decl *>(node)
      .Case<FunctionDecl>([&](auto *node) {
        this->dump(node);
      });
}

auto Dumper::dump(const Prototype *node) -> void {
  INDENT();
  llvm::errs() << "Prototype " << loc(node) << " name="<< node->name() << "\n";
}

auto Dumper::dump(const FunctionDecl *node) -> void {
  INDENT();
  llvm::errs() << "FunctionDecl " << loc(node) << "\n";
  dump(node->proto().get());
  for (auto &expr : *node) {
    dump(expr.get());
  }
}

// Expressions

auto Dumper::dump(const Expr *node) -> void {
  llvm::TypeSwitch<const Expr *>(node)
      .Case<CallExpr, DecimalExpr>([&](auto *node) {
        this->dump(node);
      })
      .Default([&](const Expr *) {
        INDENT();
        llvm::errs() << "<unknown expr, kind " << node->getKind() << ">\n";
      });
}

auto Dumper::dump(const CallExpr *node) -> void {
  INDENT();
  llvm::errs() << "CallExpr " << loc(node) << " callee=" << node->name() << "\n";
  for (auto &expr : *node) {
    dump(expr.get());
  }
}

auto Dumper::dump(const DecimalExpr *node) -> void {
  INDENT();
  llvm::errs() << "DecimalExpr " << loc(node) << " value=" << node->value() << "\n";
}

namespace cherry {

auto dumpAST(const llvm::SourceMgr &sourceManager,
             const Module &module) -> void {
  Dumper(sourceManager).dump(&module);
}

}