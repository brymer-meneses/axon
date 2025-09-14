module;

#include <algorithm>
#include <cstdint>
#include <exception>
#include <print>
#include <random>
#include <ranges>
#include <type_traits>
#include <utility>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "nanobind/ndarray.h"

export module axon.core:storage;

import axon.base;

import :data_type;

namespace nb = nanobind;

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

static auto computeShape(const nb::list& list) -> llvm::SmallVector<i64> {
  llvm::SmallVector<i64> shape;
  auto current = list;
  while (nb::isinstance<nb::list>(current) && current.size() > 0) {
    shape.push_back(static_cast<i64>(current.size()));

    if (nb::isinstance<nb::list>(current[0])) {
      current = nb::cast<nb::list>(current[0]);
    } else {
      break;
    }
  }

  return shape;
}

template <typename T>
static auto fillBuffer(const nb::list& list, T*& buffer) -> void {
  for (const auto& item : list) {
    if (nb::isinstance<nb::list>(item)) {
      fillBuffer<T>(nb::cast<nb::list>(item), buffer);
    } else {
      *buffer = nb::cast<T>(item);
      ++buffer;
    }
  }
}

export class Storage {
 public:
  static auto create(std::byte* type_erased_ptr, llvm::ArrayRef<i64> shape,
                     DataType data_type, bool is_owned,
                     Layout layout = Layout::RowMajor) -> Storage {
    auto strides = computeStrides(shape, layout);
    return {data_type, type_erased_ptr, shape, strides, is_owned};
  }

  static auto create(std::byte* type_erased_ptr, llvm::ArrayRef<i64> shape,
                     DataType data_type, bool is_owned,
                     llvm::ArrayRef<i64> strides) -> Storage {
    return {data_type, type_erased_ptr, shape, strides, is_owned};
  }

  static auto createFilled(llvm::ArrayRef<i64> shape, auto value,
                           DataType data_type, llvm::ArrayRef<i64> strides)
      -> Storage {
    auto num_elems = computeNumElems(shape);
    auto size = num_elems * data_type.getSizeInBytes();
    auto* data = new std::byte[size];

    switch (data_type.kind()) {
      case DataType::Float32: {
        auto* data_ptr = reinterpret_cast<float*>(data);
        std::fill_n(data_ptr, num_elems, value);
        break;
      }
      case DataType::Float64: {
        auto* data_ptr = reinterpret_cast<float*>(data);
        std::fill_n(data_ptr, num_elems, value);
        break;
      }
    }

    return {data_type, data, shape, strides, /*is_owned=*/true};
  }

  static auto fromNanobind(nb::ndarray<>& array, DataType data_type)
      -> Storage {
    // TODO: Explore ways to avoid copying the memory.
    auto buffer_size = array.size() * data_type.getSizeInBytes();
    auto* data_ptr = new std::byte[buffer_size];
    std::memcpy(data_ptr, array.data(), buffer_size);

    auto shape = llvm::ArrayRef<i64>(array.shape_ptr(), array.ndim());
    auto stride = llvm::ArrayRef<i64>(array.stride_ptr(), array.ndim());

    AXON_DCHECK(data_type == DataType::Float32,
                "Only float32 is supported for now.");

    return Storage::create(data_ptr, shape, data_type, /*is_owned=*/true,
                           stride);
  }

  static auto fromPython(const nb::list& list, DataType data_type) -> Storage {
    switch (data_type.kind()) {
      case DataType::Float32: {
        return fromPythonImpl<f32>(list);
      }
      case DataType::Float64: {
        return fromPythonImpl<f64>(list);
      }
    }
  }

  static auto createFilled(llvm::ArrayRef<i64> shape, auto value,
                           DataType data_type, Layout layout = Layout::RowMajor)
      -> Storage {
    auto strides = computeStrides(shape, layout);
    return createFilled(shape, value, data_type, strides);
  }

  static auto createDistributed(llvm::ArrayRef<i64> shape, float mean,
                                float standard_deviation, DataType data_type,
                                Layout layout = Layout::RowMajor) -> Storage {
    static thread_local std::mt19937 generator;

    auto num_elems = computeNumElems(shape);
    auto size_in_bytes = computeTotalSizeInBytes(shape, data_type);
    auto type_erased_ptr = new std::byte[size_in_bytes];
    auto strides = computeStrides(shape, layout);

    switch (data_type.kind()) {
      case DataType::Float32: {
        std::normal_distribution<f32> distribution(mean, standard_deviation);
        auto data_ptr = reinterpret_cast<f32*>(type_erased_ptr);
        for (int i = 0; i < num_elems; ++i) {
          data_ptr[i] = distribution(generator);
        }
        return create(type_erased_ptr, shape, data_type, /*is_owned=*/true);
      }
      case DataType::Float64: {
        std::normal_distribution<f64> distribution(mean, standard_deviation);
        auto data_ptr = reinterpret_cast<f64*>(type_erased_ptr);
        for (int i = 0; i < num_elems; ++i) {
          data_ptr[i] = distribution(generator);
        }
        return create(type_erased_ptr, shape, data_type, /*is_owned=*/true);
      }
    }
  }

  static auto createZerosLike(const Storage& storage) -> Storage {
    return createFilled(storage.shape(), 0, storage.data_type(),
                        storage.strides());
  }

  ~Storage() {
    if (is_owned_) {
      auto bytes = reinterpret_cast<std::byte*>(data_);
      delete[] bytes;
    }
  }

  Storage() = default;

  Storage(const Storage&) = delete;
  auto operator=(const Storage&) -> Storage& = delete;

  Storage(Storage&& other) {
    data_ = std::exchange(other.data_, nullptr);
    strides_ = std::move(other.strides_);
    shape_ = std::move(other.shape_);
    data_type_ = other.data_type_;
    is_owned_ = other.is_owned_;
  }

  auto operator=(Storage&& other) -> Storage& {
    if (this == &other) {
      return *this;
    }

    if (is_owned_ && data_ != nullptr) {
      delete[] data_;
    }

    data_ = std::exchange(other.data_, nullptr);
    strides_ = std::move(other.strides_);
    shape_ = std::move(other.shape_);
    data_type_ = other.data_type_;
    is_owned_ = other.is_owned_;

    return *this;
  }

  auto size() const -> size_t { return computeNumElems(shape_); }

  auto fillWithZeros() -> void {
    if (data_type_ == DataType::Float32) {
      auto* data_ptr = reinterpret_cast<f32*>(data_);
      std::fill_n(data_ptr, size(), 0.0);
    } else if (data_type_ == DataType::Float64) {
      auto* data_ptr = reinterpret_cast<f64*>(data_);
      std::fill_n(data_ptr, size(), 0.0);
    }
  }

  auto getSizeInBytes() const -> size_t {
    return size() * data_type_.getSizeInBytes();
  }

  auto data_ptr() const -> std::byte* { return data_; }

  template <typename T>
  auto as() const -> const T* {
    AXON_DCHECK(data_type_.isSameAs<T>());
    return reinterpret_cast<T*>(data_);
  }

  template <typename T>
  auto as() -> T* {
    AXON_DCHECK(data_type_.isSameAs<T>());
    return reinterpret_cast<T*>(data_);
  }

  auto shape() const -> llvm::ArrayRef<i64> { return shape_; }
  auto strides() const -> llvm::ArrayRef<i64> { return strides_; }
  auto data_type() const -> DataType { return data_type_; }

 private:
  Storage(DataType data_type, std::byte* data, llvm::ArrayRef<i64> shape,
          llvm::ArrayRef<i64> strides, bool is_owned)
      : data_(data),
        shape_(shape),
        strides_(strides),
        data_type_(data_type),
        is_owned_(is_owned) {}

  Storage(DataType data_type, std::byte* data, llvm::SmallVector<i64> shape,
          llvm::SmallVector<i64> strides, bool is_owned)
      : data_(data),
        shape_(std::move(shape)),
        strides_(std::move(strides)),
        data_type_(data_type),
        is_owned_(is_owned) {}

  template <typename T>
  static auto fromPythonImpl(const nb::list& list) -> Storage {
    auto shape = computeShape(list);
    auto total_count = computeNumElems(shape);

    auto byte_size = total_count * sizeof(T);
    auto bytes = new std::byte[byte_size];
    T* buffer = reinterpret_cast<T*>(bytes);
    T* fill_ptr = buffer;

    fillBuffer<T>(list, fill_ptr);

    return Storage::create(bytes, shape, DataType::fromType<T>(),
                           /*is_owned=*/true);
  };

  std::byte* data_;
  llvm::SmallVector<i64> shape_;
  llvm::SmallVector<i64> strides_;
  DataType data_type_;
  bool is_owned_;
};

}  // namespace axon
