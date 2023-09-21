set(LIB_HEADERS ${CHERRY_SOURCE_DIR}/include/cherry/LLVMGen)

add_library(cherryLLVMGen STATIC
        ${LIB_HEADERS}/LLVMGen.h
        LLVMGen.cpp
)

target_include_directories(cherryLLVMGen PUBLIC
        ${LIB_HEADERS}
)
