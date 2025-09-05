module;

#include <algorithm>
#include <exception>
#include <print>

#include "axon/base/dcheck.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:storage;

export namespace axon {

class DataType {
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

 private:
  InternalType type_;
};

class Storage {
 public:
  Storage(DataType element_type, std::byte* data, llvm::ArrayRef<int64_t> shape,
          llvm::ArrayRef<int64_t> strides, bool is_owned)
      : data_(data),
        shape_(shape),
        strides_(strides),
        data_type_(element_type),
        is_owned_(is_owned) {}

  Storage(const Storage&) = delete;
  auto operator=(const Storage&) -> Storage& = delete;

  Storage(Storage&& other) {
    data_ = std::exchange(other.data_, nullptr);
    strides_ = std::move(other.strides_);
    shape_ = std::move(other.shape_);
    data_type_ = other.data_type_;
    is_owned_ = other.is_owned_;
  }

  ~Storage() {
    if (is_owned_) {
      auto bytes = reinterpret_cast<std::byte*>(data_);
      delete[] bytes;
    }
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

  auto size() const -> size_t {
    int64_t num_elems = 1;
    for (int64_t dim : shape_) {
      num_elems *= dim;
    }

    return num_elems;
  }

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

  static auto createFilled(llvm::ArrayRef<int64_t> shape,
                           llvm::ArrayRef<int64_t> strides,
                           DataType element_type, auto value) -> Storage {
    int64_t num_elems = 1;
    for (auto dim : shape) {
      num_elems *= dim;
    }
    auto size = num_elems * element_type.getSizeInBytes();
    auto* data = new std::byte[size];

    if (element_type == DataType::Float32) {
      auto* data_ptr = reinterpret_cast<float*>(data);
      std::fill_n(data_ptr, num_elems, value);
    } else if (element_type == DataType::Float64) {
      auto* data_ptr = reinterpret_cast<double*>(data);
      std::fill_n(data_ptr, num_elems, value);
    }

    return {element_type, data, shape, strides, /*is_owned=*/true};
  }

  static auto createZerosLike(const Storage& storage) -> Storage {
    return createFilled(storage.shape(), storage.strides(), storage.data_type(),
                        0);
  }

  auto data_ptr() const -> std::byte* { return data_; }

  auto shape() const -> llvm::ArrayRef<int64_t> { return shape_; }
  auto strides() const -> llvm::ArrayRef<int64_t> { return strides_; }
  auto data_type() const -> DataType { return data_type_; }

 private:
  std::byte* data_;
  llvm::SmallVector<int64_t> shape_;
  llvm::SmallVector<int64_t> strides_;
  DataType data_type_;
  bool is_owned_;
};

}  // namespace axon
