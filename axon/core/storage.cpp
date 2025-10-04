module;

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <exception>
#include <memory>
#include <print>
#include <ranges>
#include <type_traits>
#include <utility>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:storage;

import axon.base;

import :data_type;
import :scalar;

namespace axon {

export enum class Layout {
  RowMajor,
  ColumnMajor,
};

export auto computeStrides(llvm::ArrayRef<i64> shape, Layout layout)
    -> llvm::SmallVector<i64> {
  llvm::SmallVector<i64> strides(shape.size());
  switch (layout) {
    case Layout::ColumnMajor: {
      i64 stride = 1;
      for (size_t i : std::views::iota(0u, shape.size())) {
        strides[i] = stride;
        stride *= shape[i];
      }
      return strides;
    }
    case Layout::RowMajor: {
      i64 stride = 1;
      for (size_t i :
           std::views::iota(0u, shape.size()) | std::views::reverse) {
        strides[i] = stride;
        stride *= shape[i];
      }
      return strides;
    }
  }
}

export auto computeNumElems(llvm::ArrayRef<i64> shape) -> i64 {
  i64 num_elems = 1;
  for (auto dim : shape) {
    num_elems *= dim;
  }
  return num_elems;
}

static auto computeTotalSizeInBytes(llvm::ArrayRef<i64> shape,
                                    DataType data_type) -> size_t {
  return computeNumElems(shape) * data_type.getSizeInBytes();
}

// Python-oriented constructors and helpers live in axon.python
export class StorageBase {
 public:
  explicit StorageBase(DataType dtype, llvm::ArrayRef<i64> shape,
                       llvm::ArrayRef<i64> strides)
      : shape_(shape.begin(), shape.end()),
        strides_(strides.begin(), strides.end()),
        dtype_(dtype) {}

  virtual ~StorageBase() = default;

  virtual auto data() -> std::byte* = 0;
  virtual auto isWritable() const -> bool = 0;
  virtual auto fillWithZeros() -> void = 0;

  auto shape() const -> llvm::ArrayRef<i64> { return shape_; }
  auto strides() const -> llvm::ArrayRef<i64> { return strides_; }
  auto data_type() const -> DataType { return dtype_; }
  auto size() const -> size_t {
    return static_cast<size_t>(computeNumElems(shape_));
  }

 protected:
  llvm::SmallVector<i64> shape_;
  llvm::SmallVector<i64> strides_;
  DataType dtype_{};
};

// Concrete storage implementations live in axon.python; this module only holds
// the type-erased interface and helpers used by factories implemented
// elsewhere.
export class Storage {
 public:
  template <typename T>
    requires std::derived_from<std::decay_t<T>, StorageBase>
  explicit Storage(T&& impl)
      : impl_(std::make_unique<std::decay_t<T>>(std::forward<T>(impl))) {}

  explicit Storage(std::unique_ptr<StorageBase> impl)
      : impl_(std::move(impl)) {}

  ~Storage() = default;

  Storage() = default;

  Storage(const Storage&) = delete;
  auto operator=(const Storage&) -> Storage& = delete;

  Storage(Storage&&) = default;
  auto operator=(Storage&&) -> Storage& = default;

  auto size() const -> size_t { return impl_ ? impl_->size() : 0; }

  auto hasValue() const -> bool { return impl_ != nullptr; }

  auto fillWithZeros() -> void {
    AXON_ASSERT(impl_, "Storage has no implementation");
    impl_->fillWithZeros();
  }

  auto getSizeInBytes() const -> size_t {
    return size() * data_type().getSizeInBytes();
  }

  auto isWritable() const -> bool { return impl_->isWritable(); }

  auto data_ptr() const -> std::byte* {
    AXON_ASSERT(impl_, "Storage has no implementation");
    return impl_->data();
  }

  template <Numeric T>
  auto as() const -> const T* {
    AXON_ASSERT(data_type().isSameAs<T>());
    return reinterpret_cast<T*>(data_ptr());
  }

  template <Numeric T>
  auto as() -> T* {
    AXON_ASSERT(data_type().isSameAs<T>());
    return reinterpret_cast<T*>(data_ptr());
  }

  auto shape() const -> llvm::ArrayRef<i64> {
    AXON_ASSERT(impl_, "Storage has no implementation");
    return impl_->shape();
  }
  auto strides() const -> llvm::ArrayRef<i64> {
    AXON_ASSERT(impl_, "Storage has no implementation");
    return impl_->strides();
  }
  auto data_type() const -> DataType {
    AXON_ASSERT(impl_, "Storage has no implementation");
    return impl_->data_type();
  }

  // Access element at given multidimensional index and return as Scalar
  auto getElementAt(llvm::ArrayRef<i64> coord) const -> Scalar {
    AXON_ASSERT(impl_, "Storage has no implementation");
    auto shp = impl_->shape();
    AXON_ASSERT(coord.size() == shp.size());

    i64 off = 0;
    auto strides = impl_->strides();
    for (size_t d = 0; d < coord.size(); ++d) {
      AXON_ASSERT(coord[d] >= 0 && coord[d] < shp[d]);
      off += coord[d] * strides[d];
    }

    switch (impl_->data_type().kind()) {
      case DataType::Float32: {
        auto base = reinterpret_cast<const f32*>(impl_->data());
        return Scalar(base[off]);
      }
      case DataType::Float64: {
        auto base = reinterpret_cast<const f64*>(impl_->data());
        return Scalar(base[off]);
      }
    }
  }

  template <Numeric T>
  auto getElementAt(llvm::ArrayRef<i64> coord) const -> T {
    AXON_ASSERT(impl_, "Storage has no implementation");
    AXON_ASSERT(data_type() == DataType::fromType<T>());
    auto shp = impl_->shape();
    AXON_ASSERT(coord.size() == shp.size());

    i64 off = 0;
    auto strides = impl_->strides();
    for (size_t d = 0; d < coord.size(); ++d) {
      AXON_ASSERT(coord[d] >= 0 && coord[d] < shp[d]);
      off += coord[d] * strides[d];
    }
    auto base = reinterpret_cast<const T*>(impl_->data());
    return base[off];
  }

 private:
  std::unique_ptr<StorageBase> impl_;
};

}  // namespace axon
