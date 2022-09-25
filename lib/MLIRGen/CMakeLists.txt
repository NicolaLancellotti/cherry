set(LIB_HEADERS ${CHERRY_HEADERS}/Cherry/MLIRGen)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_dialect_library(cherryMLIRGen
        ${LIB_HEADERS}/Conversion/ConvertCherryToArithCfFunc.h
        Conversion/ConvertCherryToArithCfFunc.cpp
        ${LIB_HEADERS}/Conversion/ConvertCherryToLLVM.h
        Conversion/ConvertCherryToLLVM.cpp
        ${LIB_HEADERS}/Conversion/ConvertCherryToSCF.h
        Conversion/ConvertCherryToSCF.cpp
        ${LIB_HEADERS}/Conversion/Passes.h
        Conversion/PassDetail.h

        ${LIB_HEADERS}/IR/CherryDialect.h
        IR/CherryDialect.cpp
        ${LIB_HEADERS}/IR/CherryOps.h
        IR/CherryOps.cpp
        IR/StructType.h

        ${LIB_HEADERS}/MLIRGen.h
        MLIRGen.cpp

        DEPENDS
        MLIRCherryOpsIncGen
        MLIRCherryConversionPassIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        ${conversion_libs}
)
