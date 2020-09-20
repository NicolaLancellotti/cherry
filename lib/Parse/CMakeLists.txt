set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/Parse)

add_library(cherryParse STATIC
        ${LIB_HEADERS}/TokenKinds.def
        ${LIB_HEADERS}/Token.h
        Token.cpp
        ${LIB_HEADERS}/Lexer.h
        Lexer.cpp
        ${LIB_HEADERS}/Parser.h
        Parser.cpp
        ${LIB_HEADERS}/DiagnosticsParse.h
        ${LIB_HEADERS}/DiagnosticsParse.def
)

target_include_directories(cherryParse PUBLIC
        ${LIB_HEADERS}
)
