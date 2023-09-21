//===--- AST.h - AST nodes and AST Dumper -----------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_AST_H
#define CHERRY_AST_H

#include "cherry/AST/Decl.h"
#include "cherry/AST/Expr.h"
#include "cherry/AST/Identifier.h"
#include "cherry/AST/Module.h"
#include "cherry/AST/Stat.h"
#include "llvm/Support/SourceMgr.h"

namespace cherry {
auto dumpAST(const llvm::SourceMgr &sourceManager, const Module &module)
    -> void;
} // end namespace cherry

#endif // CHERRY_AST_H
