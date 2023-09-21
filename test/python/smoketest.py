# RUN: %python %s | FileCheck %s

from mlir_cherry.ir import *
from mlir_cherry.dialects import builtin as builtin_d, cherry as cherry_d

with Context():
    cherry_d.register_dialect()
    module = Module.parse(
        """
    %0 = cherry.constant 10 : i64 : i64
    %1 = cherry.print %0 : (i64) -> i64
    """
    )
    # CHECK: %[[C:.*]] = cherry.constant 10 : i64
    # CHECK: cherry.print %[[C]] : (i64) -> i64
    print(str(module))
