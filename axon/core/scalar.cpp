module;

#include <array>

#include "axon/base/macros.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/HashBuilder.h"

export module axon.core:scalar;

import axon.base;
import :data_type;

namespace axon {

/// A type erased scalar value
export class Scalar {
 public:
  template <Numeric T>
  constexpr explicit Scalar(T scalar) {
    static_assert(sizeof(T) <= 8);

    data_type_ = DataType::fromType<T>();
    T* ptr = reinterpret_cast<T*>(storage_.data());
    *ptr = scalar;
  }

  template <Numeric T>
  auto as() const -> const T {
    AXON_ASSERT(DataType::fromType<T>() == data_type_,
                "Invalid passed data type.");
    return *reinterpret_cast<const T*>(storage_.data());
  }

  template <Numeric T>
  auto getAs() const -> const T {
    switch (data_type_.kind()) {
      case DataType::Float32:
        return static_cast<T>(as<f32>());
      case DataType::Float64:
        return static_cast<T>(as<f64>());
      case DataType::Int1:
        return static_cast<T>(as<bool>());
      case DataType::Int8:
        return static_cast<T>(as<i8>());
      case DataType::Int32:
        return static_cast<T>(as<i32>());
      case DataType::Int64:
        return static_cast<T>(as<i64>());
    }
    AXON_UNREACHABLE("Unsupported scalar conversion");
  }

  auto cast(DataType data_type) const -> Scalar {
    switch (data_type.kind()) {
      case DataType::Float32:
        return Scalar(getAs<f32>());
      case DataType::Float64:
        return Scalar(getAs<f64>());
      case DataType::Int1:
        return Scalar(getAs<bool>());
      case DataType::Int8:
        return Scalar(getAs<i8>());
      case DataType::Int32:
        return Scalar(getAs<i32>());
      case DataType::Int64:
        return Scalar(getAs<i64>());
    }
    AXON_UNREACHABLE("Unsupported scalar cast");
  }

  auto data_type() const -> DataType { return data_type_; }

  auto bytes() const -> const std::array<std::byte, 8>& { return storage_; }

  auto operator==(const Scalar& other) const -> bool = default;

 private:
  alignas(std::max_align_t) std::array<std::byte, 8> storage_;
  DataType data_type_;
};

export auto hash_value(const Scalar& scalar) -> llvm::hash_code {
  return llvm::hash_combine(llvm::ArrayRef<std::byte>(scalar.bytes()),
                            scalar.data_type());
}

}  // namespace axon
