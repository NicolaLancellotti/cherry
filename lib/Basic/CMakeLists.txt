set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/Basic)

add_library(cherryBasic STATIC
        ${LIB_HEADERS}/Builtins.h
        ${LIB_HEADERS}/CherryResult.h
        CherryResult.cpp
)

target_include_directories(cherryBasic PUBLIC
        ${LIB_HEADERS}
)
