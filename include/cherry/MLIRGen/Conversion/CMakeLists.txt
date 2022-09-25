set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CherryConversion)
add_public_tablegen_target(MLIRCherryConversionPassIncGen)

add_mlir_doc(Passes CherryConversionPasses ./cherry/ -gen-pass-doc)