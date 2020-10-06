//===--- AST.h - AST nodes and AST Dumper -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_AST_H
#define CHERRY_AST_H

#include "Decl.h"
#include "Expr.h"
#include "Stat.h"
#include "Module.h"
#include "llvm/Support/SourceMgr.h"

namespace cherry {
auto dumpAST(const llvm::SourceMgr &sourceManager,
             const Module &module) -> void;
} // end namespace cherry

#endif // CHERRY_AST_H
