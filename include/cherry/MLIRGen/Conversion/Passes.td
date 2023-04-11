//===--- Passes.td - Cherry passes -------------------------*- tablegen -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_PASSES_TD
#define CHERRY_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertCherryToSCF : Pass<"convert-cherry-to-scf"> {
  let summary = "Convert some Cherry ops to SCF ops";
  let constructor = "mlir::cherry::createConvertCherryToSCFPass()";
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertCherryToArithCfFunc : Pass<"convert-cherry-to-arith-cf-func"> {
  let summary = "Convert some Cherry ops to arith, cf and func ops";
  let constructor = "mlir::cherry::createConvertCherryToArithCfFunc()";
  let dependentDialects = ["arith::ArithDialect", "cf::ControlFlowDialect"];
}

def ConvertCherryToLLVM : Pass<"convert-cherry-to-llvm"> {
  let summary = "Convert Cherry ops to llvm ops";
  let constructor = "mlir::cherry::createConvertCherryToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // CHERRY_PASSES_TD
