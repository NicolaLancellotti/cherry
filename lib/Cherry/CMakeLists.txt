add_mlir_dialect_library(MLIRCherry
        CherryDialect.cpp
        CherryOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Cherry

        DEPENDS
        MLIRCherryOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
