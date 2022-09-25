get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        cherryMLIRGen
        )
add_llvm_executable(cherry-opt cherry-opt.cpp)

llvm_update_compile_flags(cherry-opt)
target_link_libraries(cherry-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(cherry-opt)
