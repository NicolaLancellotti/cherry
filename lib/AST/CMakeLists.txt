set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/AST)

add_library(cherryAST STATIC
        ${LIB_HEADERS}/AST.h
        ${LIB_HEADERS}/NodeAST.h
        ${LIB_HEADERS}/ModuleAST.h
        ${LIB_HEADERS}/DeclarationsAST.h
        ${LIB_HEADERS}/ExpressionsAST.h
        AST.cpp
)

target_include_directories(cherryAST PUBLIC
        ${LIB_HEADERS}
)