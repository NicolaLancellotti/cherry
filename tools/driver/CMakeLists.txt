add_llvm_executable(cherry-driver driver.cpp)

target_link_libraries(cherry-driver
        cherryAST
        cherryBasic
        cherryDriver
        cherryParse
        cherryLLVMGen
        cherryMLIRGen
        cherrySema
        LLVMSupport
)
