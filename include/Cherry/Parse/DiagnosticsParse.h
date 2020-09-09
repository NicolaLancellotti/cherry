//===--- DiagnosticsParse.h - Diagnostic Definitions ------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See TODO for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_DIAGNOSTICSPARSE_H
#define CHERRY_DIAGNOSTICSPARSE_H

namespace cherry {
namespace diag {
#define ERROR(ID, TEXT) const char * const ID = TEXT;
#include "DiagnosticsParse.def"
} // end namespace diag
} // end namespace cherry

#endif // CHERRY_DIAGNOSTICSPARSE_H
