//===--- DiagnosticsSema.h - Diagnostic Definitions -------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_DIAGNOSTICSSEMA_H
#define CHERRY_DIAGNOSTICSSEMA_H

namespace cherry {
namespace diag {
#define ERROR(ID, TEXT) const char *const ID = TEXT;
#include "DiagnosticsSema.def"

} // end namespace diag
} // end namespace cherry

#endif // CHERRY_DIAGNOSTICSSEMA_H
