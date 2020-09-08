add_llvm_executable(cherry-driver driver.cpp)

target_link_libraries(cherry-driver
        LLVMSupport
        cherryBasic
        cherryDriver
        cherryParse
        cherryAST
        cherryIRGen
)