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

### Driver Flags
Meaning                             |  Flag
|-----------------------------------|-------------------|
Dump tokens 			                  | -dump=tokens
Dump the AST                        | -dump=ast
Dump the MLIR (cherry)              | -dump=mlir
Dump the MLIR (standard)            | -dump=mlir-std
Dump the MLIR (LLVM)                | -dump=mlir-llvm
Dump the LLVM IR                    | -dump=llvm
Select the LLVM backend             | -b=llvm
Select the MLIR backend (default)   | -b=mlir
Parse and type-check                | -typecheck
Enable optimisation                 | -opt
Generate a target ".o" object file  | -c[=\<filename>]  

## Examples
main.cherry:
```
# This is a comment

struct A { }

struct B {
  x: UInt64,
  y: A
}

fun baz(x: B) { }

fun bar(x: UInt64, y: UInt64) {
  print(x);
  print(y);
  print(18446744073709551615);
}

fun main() {
  bar(0, 1);
}
```

### Run JIT
run:
```
cherry-driver main.cherry
```
output:
```
0
1
18446744073709551615
```

### Generate an object file and run
run:
```
cherry-driver main.cherry -c=a.o

clang a.o

./a.out 
```
output:
```
0
1
18446744073709551615
```

## Run the Optimiser
Example
```
cherry-opt -lower-cherry-to-std -lower-cherry-std-to-llvm -print-ir-after-all main.mlir
```

## Unimplemented features
- Struct constructors and struct access expressions 
can be parsed and type-checked but the lowering 
to MLIR or LLVM is not implemented yet.
- JIT is available only with the MLIR backend
