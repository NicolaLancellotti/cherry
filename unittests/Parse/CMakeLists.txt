add_cherry_unittest(CherryParseTests
        ../TestRunner.cpp
        LexerTest.cpp
)

target_link_libraries(CherryParseTests
        PRIVATE
        cherryParse
)
