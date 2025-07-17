module;

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>

#include "axon/base/dcheck.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"

export module axon.core:storage;

namespace axon {

export enum class DataType {
  Float32,
  Float64,
  Int32,
  Int64,
};

template <typename T>
consteval auto type_to_dtype() -> DataType {
  if (std::is_same_v<float, T>) {
    return DataType::Float32;
  } else if (std::is_same_v<double, T>) {
    return DataType::Float64;
  } else if (std::is_same_v<int32_t, T>) {
    return DataType::Int32;
  } else if (std::is_same_v<int64_t, T>) {
    return DataType::Int64;
  }

  static_assert(false, "Unsupported data type");
}

/// The underlying storage of a tensor.
export class Storage {
 public:
  template <typename T>
  Storage(llvm::SmallVector<int64_t> shape) {
    dtype_ = type_to_dtype<T>();
    const auto total_size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

    data_ = new T[total_size];
    shape_ = std::move(shape);
  }

  auto shape() const -> llvm::ArrayRef<int64_t> { return shape_; }

  template <typename T>
  auto adapt() -> xt::xarray<T> {
    AXON_DCHECK(type_to_dtype<T>() == dtype_, "Invalid type T.");
    return xt::adapt(static_cast<T*>(data_), xt::no_ownership(), shape_);
  }

  ~Storage() { delete[] data_; }

  Storage(const Storage&) = delete;
  auto operator=(const Storage&) -> void = delete;

  Storage(Storage&& other) {
    dtype_ = other.dtype_;
    data_ = other.data_;
    shape_ = std::move(other.shape_);
    other.data_ = nullptr;
  }

  auto operator=(Storage&& other) -> void {
    dtype_ = other.dtype_;
    data_ = other.data_;
    shape_ = std::move(other.shape_);
    other.data_ = nullptr;
  }

 private:
  DataType dtype_;
  std::byte* data_;
  llvm::SmallVector<int64_t> shape_;
};

}  // namespace axon
