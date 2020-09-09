set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/Basic)

add_library(cherryBasic STATIC
        ${LIB_HEADERS}/CherryResult.h
        CherryResult.cpp
)

target_include_directories(cherryBasic PUBLIC
        ${LIB_HEADERS}
)
