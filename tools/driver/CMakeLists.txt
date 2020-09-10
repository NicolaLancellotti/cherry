add_llvm_executable(cherry-driver driver.cpp)

target_link_libraries(cherry-driver
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSPIRV
        MLIRTranslation
        MLIRStandardToLLVM
        MLIRSupport
        LLVMSupport
        cherryBasic
        cherryDriver
        cherryParse
        cherryAST
        cherryIRGen
)