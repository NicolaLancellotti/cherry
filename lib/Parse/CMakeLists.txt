set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/Parse)

add_library(cherryParse STATIC
        ${LIB_HEADERS}/Token.h
        ${LIB_HEADERS}/TokenKinds.def
        ${LIB_HEADERS}/Lexer.h
        Lexer.cpp
        Token.cpp
)

target_include_directories(cherryParse PUBLIC
        ${LIB_HEADERS}
)