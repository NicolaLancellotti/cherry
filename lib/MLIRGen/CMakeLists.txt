set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/MLIRGen)

add_mlir_dialect_library(cherryMLIRGen
		${LIB_HEADERS}/MLIRGen.h
		MLIRGen.cpp

        CherryDialect.cpp
        CherryOps.cpp

		${LIB_HEADERS}/Passes.h
		LowerToStandardPass.cpp
		LowerToLLVMPass.cpp

		StructType.h

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/cherry/MLIRGen

        DEPENDS
        MLIRCherryOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
)
