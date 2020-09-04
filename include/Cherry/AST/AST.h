#ifndef CHERRY_AST_H
#define CHERRY_AST_H

#include "cherry/AST/Module.h"
#include "cherry/AST/Declarations.h"
#include "cherry/AST/Expressions.h"
#include "llvm/Support/SourceMgr.h"

namespace cherry {
auto dumpAST(const llvm::SourceMgr &sourceManager,
             const Module &module) -> void;
}

#endif
