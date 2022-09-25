//===--- Identifier.h - Cherry Language Identifier ASTs ---------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_IDENTIFIER_H
#define CHERRY_IDENTIFIER_H

#include "Node.h"
#include "llvm/ADT/StringRef.h"

namespace cherry {

class Identifier : public Node {
public:
  explicit Identifier(llvm::SMLoc location, llvm::StringRef name)
      : Node{location}, _name(name.str()){};

  auto name() const -> llvm::StringRef { return _name; }

private:
  std::string _name;
};

class Type final : public Identifier {
public:
  using Identifier::Identifier;
};

class FunctionName final : public Identifier {
public:
  using Identifier::Identifier;
};

} // end namespace cherry

#endif // CHERRY_IDENTIFIER_H
