# 🍒 Cherry Programming Language 🍒

## Building Cherry

- Built LLVM and MLIR in `$BUILD_DIR` and instal them to `$PREFIX`.
  - Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

- To build Cherry and launch the tests, run:
```sh
mkdir build && cd build
cmake -G "Unix Makefiles" .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-cherry
```
- To build the documentation from the TableGen description of the dialect operations, run:
```sh
cmake --build . --target mlir-doc
```

## Grammar
[Cherry grammar](/docs/Grammar.md)

## Flags

Meaning                        					 |  Flag
|------------------------------------------------|-------------------|
Enable optimisation                              | -opt
Dump tokens to the standard error 			     | -dump=tokens
Dump the AST to the standard error               | -dump=ast
Dump the MLIR to the standard error              | -dump=mlir
