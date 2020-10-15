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
Dump tokens 			            | -dump=tokens
Parse and dump the AST              | -dump=parse
Parse, type-check and dump the AST  | -dump=ast
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
### Syntax example
```
# This is a comment

struct A { }

struct B {
  x: Bool,
  y: A
}

fn bar(x: UInt64, y: Bool): B {
  var k: Bool = y;
  
  k = if k {
    print(18446744073709551615);
    false
  } else {
    print(0);
    true
  };

  var unit: () = while k {
  	k = false;
  	()
  };

  B(k, A())
}

fn baz(): () {
  ()
}

fn main(): UInt64 {
  0 % 3 * 8 / 4 + 3 - 1;
  3 lt 1; 3 le 1; 3 gt 1; 3 ge 1;
  true and false or true eq false neq true;

  var structValue: B = bar(18446744073709551615, false);
  print(boolToUInt64(structValue.x));
  baz();
  1
}
```

### Run JIT
```
cherry-driver main.cherry
```

### Generate an object file and run
```
cherry-driver main.cherry -c=a.o

clang a.o

./a.out 
```

## Run the Optimiser
Example
```
cherry-opt -lower-cherry-to-std -lower-cherry-std-to-llvm -print-ir-after-all main.mlir
```

## Unimplemented features
- Structs are implemented only in LLVMGen (-b=llvm), 
in MLIRGen (-b=mlir) they are not implemented yet.