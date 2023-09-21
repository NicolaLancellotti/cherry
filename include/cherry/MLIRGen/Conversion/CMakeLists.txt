set(LLVM_TARGET_DEFINITIONS CherryPasses.td)
mlir_tablegen(CherryPasses.h.inc -gen-pass-decls -name CherryConversion)
add_public_tablegen_target(MLIRCherryConversionPassIncGen)

add_mlir_doc(CherryPasses CherryConversionPasses ./cherry/ -gen-pass-doc)
