add_llvm_executable(cherry-driver driver.cpp)

target_link_libraries(cherry-driver
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslation
        MLIRSPIRV
        MLIRStandardToLLVM
        MLIRSCFToStandard
        MLIRSupport
        LLVMSupport
        cherryAST
        cherryBasic
        cherryDriver
        cherryParse
        cherryLLVMGen
        cherryMLIRGen
        cherrySema
)