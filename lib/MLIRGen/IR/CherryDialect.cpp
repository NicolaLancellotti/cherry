//===- CherryDialect.cpp - Cherry dialect ---------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/CherryDialect.h"
#include "cherry/MLIRGen/IR/CherryOps.h"
#include "StructType.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::cherry;

#include "cherry/MLIRGen/IR/CherryOpsDialect.cpp.inc"

void CherryDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cherry/MLIRGen/IR/CherryOps.cpp.inc"
      >();
  addTypes<StructType>();
}

//   struct-type ::= `struct` `<` type (`,` type)* `>`
auto CherryDialect::parseType(mlir::DialectAsmParser &parser) const
    -> mlir::Type {
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  SmallVector<mlir::Type, 1> elementTypes;
  do {
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    if (!elementType.isa<IntegerType, StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a integer or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseGreater())
    return Type();

  return StructType::get(elementTypes);
}

auto CherryDialect::printType(mlir::Type type,
                              mlir::DialectAsmPrinter &printer) const -> void {
  StructType structType = type.cast<StructType>();
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}