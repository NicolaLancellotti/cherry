set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/AST)

add_library(cherryAST STATIC
        ${LIB_HEADERS}/AST.h
        ${LIB_HEADERS}/Node.h
        ${LIB_HEADERS}/Module.h
        ${LIB_HEADERS}/Decl.h
        ${LIB_HEADERS}/Expr.h
        ${LIB_HEADERS}/Stat.h
        ASTDumper.cpp
)

target_include_directories(cherryAST PUBLIC
        ${LIB_HEADERS}
)
