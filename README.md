# 🍒 Cherry Programming Language 🍒
## Dependencies
Install the following dependencies:
- CMake
- Ninja
- Make
- Clang

## Build Cherry
To build the LLVM project and Cherry run:
```sh
make all
```
## Grammar & Builtins
[Cherry grammar](/docs/Grammar.md)

[Builtins](/docs/Builtins.md)

## Driver Flags
*See test/cherry/Driver for invocation examples.*

Meaning                                           |  Flag
|-------------------------------------------------|-------------------|
Dump tokens                                       | -dump=tokens
Parse and dump the AST                            | -dump=parse
Parse, type-check and dump the AST                | -dump=ast
Dump the MLIR (cherry)                            | -dump=mlir
Dump the MLIR (cherry + scf)                      | -dump=mlir1
Dump the MLIR (cherry + scf + arith + cf + func)  | -dump=mlir2
Dump the MLIR (LLVM)                              | -dump=mlir-llvm
Dump the LLVM IR                                  | -dump=llvm
Select the LLVM backend                           | -b=llvm
Select the MLIR backend (default)                 | -b=mlir
Parse and type-check                              | -typecheck
Enable optimisation                               | -opt
Generate a target ".o" object file                | -c[=\<FILE_PATH>]  

## Syntax Example 
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
    print(1);
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
  0
}
```
