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
    AXON_DCHECK(DataType::fromType<T>() == data_type_,
                "Invalid passed data type.");
    return *reinterpret_cast<const T*>(storage_.data());
  }

  friend auto operator+(const Scalar& lhs, const Scalar& rhs) -> Scalar {
    AXON_DCHECK(lhs.data_type() == rhs.data_type());
    switch (lhs.data_type().kind()) {
      case DataType::Float32: {
        auto lhs_casted = lhs.as<f32>();
        auto rhs_casted = rhs.as<f32>();
        return Scalar(lhs_casted + rhs_casted);
      }
      case DataType::Float64: {
        auto lhs_casted = lhs.as<f64>();
        auto rhs_casted = rhs.as<f64>();
        return Scalar(lhs_casted + rhs_casted);
      }
    }
  }

  template <Numeric T>
  friend auto operator-(const Scalar& lhs, T rhs) -> Scalar {
    AXON_DCHECK(lhs.data_type() == DataType::fromType<T>());
    switch (lhs.data_type().kind()) {
      case DataType::Float32: {
        auto lhs_casted = lhs.as<f32>();
        return Scalar(lhs_casted - rhs);
      }
      case DataType::Float64: {
        auto lhs_casted = lhs.as<f64>();
        return Scalar(lhs_casted - rhs);
      }
    }
  }

  template <Numeric T>
  friend auto operator-(T lhs, const Scalar& rhs) -> Scalar {
    AXON_DCHECK(rhs.data_type() == DataType::fromType<T>());
    switch (rhs.data_type().kind()) {
      case DataType::Float32: {
        auto rhs_casted = rhs.as<f32>();
        return Scalar(lhs - rhs_casted);
      }
      case DataType::Float64: {
        auto rhs_casted = rhs.as<f64>();
        return Scalar(lhs - rhs_casted);
      }
    }
  }

  template <Numeric T>
  friend auto operator+(const Scalar& lhs, T rhs) -> Scalar {
    AXON_DCHECK(lhs.data_type() == DataType::fromType<T>());
    switch (lhs.data_type().kind()) {
      case DataType::Float32: {
        auto lhs_casted = lhs.as<f32>();
        return Scalar(lhs_casted + rhs);
      }
      case DataType::Float64: {
        auto lhs_casted = lhs.as<f64>();
        return Scalar(lhs_casted + rhs);
      }
    }
  }

  template <Numeric T>
  friend auto operator+(T lhs, const Scalar& rhs) -> Scalar {
    AXON_DCHECK(rhs.data_type() == DataType::fromType<T>());
    switch (rhs.data_type().kind()) {
      case DataType::Float32: {
        auto rhs_casted = rhs.as<f32>();
        return Scalar(lhs + rhs_casted);
      }
      case DataType::Float64: {
        auto rhs_casted = rhs.as<f64>();
        return Scalar(lhs + rhs_casted);
      }
    }
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
