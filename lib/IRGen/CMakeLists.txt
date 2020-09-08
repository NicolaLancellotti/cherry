set(LIB_HEADERS ${CHERRY_HEADERS}/cherry/IRGen)

add_mlir_dialect_library(cherryIRGen
		${LIB_HEADERS}/MLIRGen.h
		MLIRGen.cpp

        CherryDialect.cpp
        CherryOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/IRGen

        DEPENDS
        MLIRCherryOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
