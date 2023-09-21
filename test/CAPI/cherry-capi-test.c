//===- cherry-capo-demo.c - Simple demo of C-API --------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

// RUN: cherry-capi-test 2>&1 | FileCheck %s

#include "cherry/MLIRGen/Cherry-c/Dialects.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include <stdio.h>

static void registerAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  // TODO: Create the dialect handles for the builtin dialects and avoid this.
  // This adds dozens of MB of binary size over just the cherry dialect.
  registerAllUpstreamDialects(ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__cherry__(), ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString("%0 = cherry.constant 10 : i64 : i64\n"
                                          "%1 = cherry.print %0 : (i64) -> i64"));
  if (mlirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }
  MlirOperation op = mlirModuleGetOperation(module);

  // CHECK: %[[C:.*]] = cherry.constant 10 : i64
  // CHECK: cherry.print %[[C]] : (i64) -> i64
  mlirOperationDump(op);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return 0;
}
