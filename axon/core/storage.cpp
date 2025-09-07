module;

#include <algorithm>
#include <exception>
#include <print>
#include <random>
#include <ranges>
#include <utility>

#include "axon/base/dcheck.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:storage;

namespace axon {

export class DataType {
 public:
  enum InternalType {
    Float32,
    Float64,
  };

  DataType(InternalType type) : type_(type) {}
  DataType() = default;

  auto operator==(DataType other) const -> bool { return type_ == other.type_; }

  auto getSizeInBytes() const -> size_t {
    switch (type_) {
      case DataType::Float32:
        return 4;
      case DataType::Float64:
        return 8;
    }
  }

  auto value() -> InternalType { return type_; }

 private:
  InternalType type_;
};

export enum class Layout {
  RowMajor,
  ColumnMajor,
};

static auto computeStrides(llvm::ArrayRef<int64_t> shape, Layout layout)
    -> llvm::SmallVector<int64_t> {
  llvm::SmallVector<int64_t> strides(shape.size());
  switch (layout) {
    case Layout::ColumnMajor: {
      int64_t stride = 1;
      for (size_t i : std::views::iota(0u, shape.size())) {
        strides[i] = stride;
        stride *= shape[i];
      }
      return strides;
    }
    case Layout::RowMajor: {
      int64_t stride = 1;
      for (size_t i :
           std::views::iota(0u, shape.size()) | std::views::reverse) {
        strides[i] = stride;
        stride *= shape[i];
      }
      return strides;
    }
  }
}

static auto computeNumElems(llvm::ArrayRef<int64_t> shape) -> int64_t {
  int64_t num_elems = 1;
  for (auto dim : shape) {
    num_elems *= dim;
  }
  return num_elems;
}

static auto computeTotalSizeInBytes(llvm::ArrayRef<int64_t> shape,
                                    DataType data_type) -> size_t {
  return computeNumElems(shape) * data_type.getSizeInBytes();
}

export class Storage {
 public:
  static auto create(std::byte* type_erased_ptr, llvm::ArrayRef<int64_t> shape,
                     DataType data_type, bool is_owned,
                     Layout layout = Layout::RowMajor) -> Storage {
    auto strides = computeStrides(shape, layout);
    return {data_type, type_erased_ptr, shape, strides, is_owned};
  }

  static auto create(std::byte* type_erased_ptr, llvm::ArrayRef<int64_t> shape,
                     DataType data_type, bool is_owned,
                     llvm::ArrayRef<int64_t> strides) -> Storage {
    return {data_type, type_erased_ptr, shape, strides, is_owned};
  }

  static auto createFilled(llvm::ArrayRef<int64_t> shape, auto value,
                           DataType data_type, llvm::ArrayRef<int64_t> strides)
      -> Storage {
    auto num_elems = computeNumElems(shape);
    auto size = num_elems * data_type.getSizeInBytes();
    auto* data = new std::byte[size];

    switch (data_type.value()) {
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

  static auto createFilled(llvm::ArrayRef<int64_t> shape, auto value,
                           DataType data_type, Layout layout = Layout::RowMajor)
      -> Storage {
    auto strides = computeStrides(shape, layout);
    return createFilled(shape, value, data_type, strides);
  }

  static auto createDistributed(llvm::ArrayRef<int64_t> shape, float mean,
                                float standard_deviation, DataType data_type,
                                Layout layout = Layout::RowMajor) -> Storage {
    static thread_local std::mt19937 generator;

    auto num_elems = computeNumElems(shape);
    auto size_in_bytes = computeTotalSizeInBytes(shape, data_type);
    auto type_erased_ptr = new std::byte[size_in_bytes];
    auto strides = computeStrides(shape, layout);

    switch (data_type.value()) {
      case DataType::Float32: {
        std::normal_distribution<float> distribution(mean, standard_deviation);
        auto data_ptr = reinterpret_cast<float*>(type_erased_ptr);
        for (int i = 0; i < num_elems; ++i) {
          data_ptr[i] = distribution(generator);
        }
        return create(type_erased_ptr, shape, data_type, /*is_owned=*/true);
      }
      case DataType::Float64: {
        std::normal_distribution<double> distribution(mean, standard_deviation);
        auto data_ptr = reinterpret_cast<double*>(type_erased_ptr);
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
      auto* data_ptr = reinterpret_cast<float*>(data_);
      std::fill_n(data_ptr, size(), 0.0);
    } else if (data_type_ == DataType::Float64) {
      auto* data_ptr = reinterpret_cast<double*>(data_);
      std::fill_n(data_ptr, size(), 0.0);
    }
  }

  auto getSizeInBytes() const -> size_t {
    return size() * data_type_.getSizeInBytes();
  }

  auto data_ptr() const -> std::byte* { return data_; }

  auto shape() const -> llvm::ArrayRef<int64_t> { return shape_; }
  auto strides() const -> llvm::ArrayRef<int64_t> { return strides_; }
  auto data_type() const -> DataType { return data_type_; }

 private:
  Storage(DataType element_type, std::byte* data, llvm::ArrayRef<int64_t> shape,
          llvm::ArrayRef<int64_t> strides, bool is_owned)
      : data_(data),
        shape_(shape),
        strides_(strides),
        data_type_(element_type),
        is_owned_(is_owned) {}

  Storage(DataType element_type, std::byte* data,
          llvm::SmallVector<int64_t> shape, llvm::SmallVector<int64_t> strides,
          bool is_owned)
      : data_(data),
        shape_(std::move(shape)),
        strides_(std::move(strides)),
        data_type_(element_type),
        is_owned_(is_owned) {}

  std::byte* data_;
  llvm::SmallVector<int64_t> shape_;
  llvm::SmallVector<int64_t> strides_;
  DataType data_type_;
  bool is_owned_;
};

}  // namespace axon
