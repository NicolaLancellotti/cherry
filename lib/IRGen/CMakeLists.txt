set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/IRGen)

add_mlir_dialect_library(cherryIRGen
		${LIB_HEADERS}/MLIRGen.h
		MLIRGen.cpp

        CherryDialect.cpp
        CherryOps.cpp

		${LIB_HEADERS}/Passes.h
		LowerToStandardPass.cpp
		LowerToLLVMPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/cherry/IRGen

        DEPENDS
        MLIRCherryOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
)
