set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/LLVMGen)

add_library(cherryLLVMGen STATIC
        ${LIB_HEADERS}/LLVMGen.h
        LLVMGen.cpp
)

target_include_directories(cherryLLVMGen PUBLIC
        ${LIB_HEADERS}
)
