set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/Driver)

add_library(cherryDriver STATIC
        ${LIB_HEADERS}/Compilation.h
        Compilation.cpp
)

target_include_directories(cherryDriver PUBLIC
        ${LIB_HEADERS}
)
