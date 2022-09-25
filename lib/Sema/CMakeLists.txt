set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/Sema)

add_library(cherrySema STATIC
        ${LIB_HEADERS}/Sema.h
        Sema.cpp
        Symbols.h
        ${LIB_HEADERS}/DiagnosticsSema.h
        ${LIB_HEADERS}/DiagnosticsSema.def
)

target_include_directories(cherrySema PUBLIC
        ${LIB_HEADERS}
)
