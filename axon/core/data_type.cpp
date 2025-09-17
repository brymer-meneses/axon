module;

#include <cstdint>
#include <type_traits>

#include "llvm/ADT/Hashing.h"

export module axon.core:data_type;

import axon.base;

namespace axon {

export class DataType {
 public:
  enum InternalType : u8 {
    Float32,
    Float64,
  };

  constexpr DataType(InternalType type) : type_(type) {}
  constexpr DataType() = default;

  auto operator==(DataType other) const -> bool { return type_ == other.type_; }

  auto getSizeInBytes() const -> size_t {
    switch (type_) {
      case DataType::Float32:
        return 4;
      case DataType::Float64:
        return 8;
    }
  }

  auto kind() const -> InternalType { return type_; }

  template <Numeric T>
  auto isSameAs() const -> bool {
    return fromType<T>() == type_;
  }

  template <Numeric T>
  static consteval auto fromType() -> DataType {
    if constexpr (std::is_same_v<T, f32>) {
      return DataType::Float32;
    } else if constexpr (std::is_same_v<T, f64>) {
      return DataType::Float64;
    } else {
      static_assert("Passed template parameter has no corresponding DataType");
    }
  }

 private:
  InternalType type_;
};

export auto hash_value(const DataType& data_type) -> llvm::hash_code {
  return llvm::hash_combine(data_type.kind());
}

}  // namespace axon
