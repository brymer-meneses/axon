#pragma once

#include <cstdint>
#include <tuple>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/HashBuilder.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect
#include "Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "DialectTypeDefs.h.inc"

namespace axon {

struct ParameterTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<llvm::ArrayRef<int64_t>, mlir::Type>;

  ParameterTypeStorage(llvm::ArrayRef<int64_t> shape, mlir::Type type)
      : shape(shape), type(type) {}

  auto operator==(const KeyTy& key) const -> bool {
    return key == std::make_pair(shape, type);
  }

  static auto hashKey(const KeyTy& key) -> llvm::hash_code {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(llvm::ArrayRef<int64_t> shape, mlir::Type type) {
    return KeyTy(std::make_pair(shape, type));
  }

  static auto construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
      -> ParameterTypeStorage*;

  llvm::ArrayRef<int64_t> shape;
  mlir::Type type;
};

class ParameterType : public mlir::Type::TypeBase<ParameterType, mlir::Type,
                                                  ParameterTypeStorage> {
 public:
  using Base::Base;
  /// The name of this struct type.
  static constexpr llvm::StringLiteral name = "toy.struct";

  static ParameterType get(llvm::ArrayRef<int64_t> shape, mlir::Type type);

  llvm::ArrayRef<int64_t> getShape() const;
  mlir::Type getElementType() const;
};

}  // namespace axon
