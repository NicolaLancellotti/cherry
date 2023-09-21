// RUN: mlir-opt %s --load-dialect-plugin=%cherry_libs/CherryPlugin%shlibext --pass-pipeline="builtin.module(convert-cherry-to-arith-cf-func)" | FileCheck %s

module  {
  func.func @main() -> i64 {
    // CHECK-LABEL:  arith.constant 10 : i64
    %0 = "cherry.constant"() {value = 10 : i64} : () -> i64
    "cherry.return"(%0) : (i64) -> ()
  }
}
