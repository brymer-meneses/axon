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
