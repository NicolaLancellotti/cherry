#ifndef CHERRY_DIAGNOSTICSPARSE_H
#define CHERRY_DIAGNOSTICSPARSE_H

namespace cherry {
namespace diag {
#define ERROR(ID, TEXT) const char * const ID = TEXT;
#include "DiagnosticsParse.def"
}
}

#endif
