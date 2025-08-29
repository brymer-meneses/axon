module;

#include <algorithm>
#include <exception>
#include <print>

#include "axon/base/dcheck.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:storage;

export namespace axon {

class ElementType {
 public:
  enum InternalType {
    Float32,
    Float64,
  };

  ElementType(InternalType type) : type_(type) {}
  ElementType() = default;

  auto operator==(ElementType other) const -> bool {
    return type_ == other.type_;
  }

  auto getSizeInBytes() const -> size_t {
    switch (type_) {
      case ElementType::Float32:
        return 4;
      case ElementType::Float64:
        return 8;
    }
  }

 private:
  InternalType type_;
};

class Storage {
 public:
  Storage(ElementType element_type, std::byte* data,
          llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
          bool is_owned)
      : data_(data),
        shape_(shape),
        strides_(strides),
        element_type_(element_type),
        is_owned_(is_owned) {}

  Storage(const Storage&) = delete;
  auto operator=(const Storage&) -> Storage& = delete;

  Storage(Storage&& other) {
    data_ = std::exchange(other.data_, nullptr);
    strides_ = std::move(other.strides_);
    shape_ = std::move(other.shape_);
    element_type_ = other.element_type_;
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
    element_type_ = other.element_type_;
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
    if (element_type_ == ElementType::Float32) {
      auto* data_ptr = reinterpret_cast<float*>(data_);
      std::fill_n(data_ptr, size(), 0.0);
    } else if (element_type_ == ElementType::Float64) {
      auto* data_ptr = reinterpret_cast<double*>(data_);
      std::fill_n(data_ptr, size(), 0.0);
    }
  }

  auto getSizeInBytes() const -> size_t {
    return size() * element_type_.getSizeInBytes();
  }

  static auto createFilled(llvm::ArrayRef<int64_t> shape,
                           llvm::ArrayRef<int64_t> strides,
                           ElementType element_type, auto value) -> Storage {
    int64_t num_elems = 1;
    for (auto dim : shape) {
      num_elems *= dim;
    }
    auto size = num_elems * element_type.getSizeInBytes();
    auto* data = new std::byte[size];

    if (element_type == ElementType::Float32) {
      auto* data_ptr = reinterpret_cast<float*>(data);
      std::fill_n(data_ptr, size, value);
    } else if (element_type == ElementType::Float64) {
      auto* data_ptr = reinterpret_cast<double*>(data);
      std::fill_n(data_ptr, size, value);
    }

    return {element_type, data, shape, strides, /*is_owned=*/true};
  }

  static auto createZerosLike(const Storage& storage) -> Storage {
    return createFilled(storage.shape(), storage.strides(),
                        storage.element_type(), 0);
  }

  auto data_ptr() const -> std::byte* { return data_; }

  auto shape() const -> llvm::ArrayRef<int64_t> { return shape_; }
  auto strides() const -> llvm::ArrayRef<int64_t> { return strides_; }
  auto element_type() const -> ElementType { return element_type_; }

 private:
  std::byte* data_;
  llvm::SmallVector<int64_t> shape_;
  llvm::SmallVector<int64_t> strides_;
  ElementType element_type_;
  bool is_owned_;
};

}  // namespace axon
