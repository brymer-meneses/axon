module;

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "nanobind/ndarray.h"

export module axon.python:storage;

import std;

namespace nb = nanobind;

export namespace axon {

enum class ElementType {
  Float32,
  Float64,
};

auto elementTypeSize(ElementType type) -> int64_t {
  switch (type) {
    case ElementType::Float32:
      return 4;
    case ElementType::Float64:
      return 8;
  }
}

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
    for (auto dim : shape_) {
      num_elems *= dim;
    }
    return num_elems * elementTypeSize(element_type_);
  }

  static auto fromNanobind(nb::ndarray<>& array, ElementType element_type)
      -> Storage {
    auto* data = reinterpret_cast<std::byte*>(array.data());
    auto shape = llvm::ArrayRef<int64_t>(array.shape_ptr(), array.ndim());
    auto stride = llvm::ArrayRef<int64_t>(array.stride_ptr(), array.ndim());

    return {element_type, data, shape, stride, /*is_owned=*/false};
  }

  static auto createZerosLike(const Storage& storage) -> Storage {
    auto size = storage.size();
    auto* data = new std::byte[size];
    std::memset(data, 0, size);

    return {storage.element_type_, data, storage.shape_, storage.strides_,
            /*is_owned=*/true};
  }

 private:
  std::byte* data_;
  llvm::SmallVector<int64_t> shape_;
  llvm::SmallVector<int64_t> strides_;
  ElementType element_type_;
  bool is_owned_;
};

}  // namespace axon
