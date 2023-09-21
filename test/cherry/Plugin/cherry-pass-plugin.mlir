// RUN: mlir-opt %s --load-pass-plugin=%cherry_libs/CherryPlugin%shlibext --pass-pipeline="builtin.module(convert-cherry-to-llvm)" | FileCheck %s

module {
  func.func @main() -> i64 {
    // CHECK-LABEL: llvm.mlir.constant(10 : i64) : i64
    %c10_i64 = arith.constant 10 : i64
    return %c10_i64 : i64
  }
}
