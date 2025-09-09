
module;

#include <variant>

#include "axon/base/dcheck.h"
#include "type_traits"

export module axon.core:scalar;

import axon.base;
import :storage;

namespace axon {

/// A type erased scalar value
export class Scalar {
 public:
  template <typename T>
  Scalar(T scalar) {
    static_assert(sizeof(T) <= 8);

    data_type_ = DataType::fromType<T>();
    T* ptr = reinterpret_cast<T*>(storage_.data());
    *ptr = scalar;
  }

  template <typename T>
  auto as() const -> const T {
    AXON_DCHECK(DataType::fromType<T>() == data_type_,
                "Invalid passed data type.");
    return *reinterpret_cast<const T*>(storage_.data());
  }

  auto data_type() const -> DataType { return data_type_; }

 private:
  alignas(std::max_align_t) std::array<std::byte, 8> storage_;
  DataType data_type_;
};

}  // namespace axon
