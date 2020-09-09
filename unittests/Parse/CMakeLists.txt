add_cherry_unittest(CherryParseTests
        LexerTest.cpp
)

target_link_libraries(CherryParseTests
        PRIVATE
        cherryParse
)
