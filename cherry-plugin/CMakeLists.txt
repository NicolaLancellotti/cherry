add_llvm_library(CherryPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        cherry-plugin.cpp

        DEPENDS
        cherryMLIRGen
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        cherryMLIRGen
        )
target_include_directories(
        CherryPlugin
        PRIVATE
        "${CHERRY_BINARY_DIR}/include"
)
