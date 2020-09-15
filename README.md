# 🍒 Cherry Programming Language 🍒

## Build Cherry

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
## Grammar & Builtins
[Cherry grammar](/docs/Grammar.md)
[Builtins](/docs/Builtins.md)

## Run the Driver

### Run JIT
Example
```
cherry-driver main.cherry
```

### Driver Flags
Meaning                    |  Flag
|--------------------------|-------------------|
Enable optimisation        | -opt
Dump tokens 			   | -dump=tokens
Dump the AST               | -dump=ast
Dump the MLIR (cherry)     | -dump=mlir
Dump the MLIR (standard)   | -dump=mlir-std
Dump the MLIR (LLVM)       | -dump=mlir-llvm
Dump the LLVM IR           | -dump=llvm

Example
```
cherry-driver -dump=mlir main.cherry
```

## Run the Optimiser
Example
```
cherry-opt -lower-cherry-to-std -lower-cherry-std-to-llvm -print-ir-after-all main.mlir
```