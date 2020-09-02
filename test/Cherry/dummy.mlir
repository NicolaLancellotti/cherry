// RUN: cherry-opt %s | cherry-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = cherry.foo %{{.*}} : i32
        %res = cherry.foo %0 : i32
        return
    }
}
