module;

#include <algorithm>
#include <memory>
#include <random>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"

export module axon.python:storage;

import axon.base;
import axon.core;

namespace nb = nanobind;

namespace axon {

// Helper functions previously in core
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

template <Numeric T>
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

// Concrete implementations that satisfy the Storage concept
export class CpuStorage final : public StorageBase {
 public:
  CpuStorage(std::byte* data, llvm::ArrayRef<i64> shape,
             llvm::ArrayRef<i64> strides, DataType dtype, bool is_owned)
      : StorageBase(dtype, shape, strides), data_(data), is_owned_(is_owned) {}

  CpuStorage(llvm::ArrayRef<i64> shape, llvm::ArrayRef<i64> strides,
             DataType data_type_)
      : StorageBase(data_type_, shape, strides) {
    auto size_in_bytes = computeNumElems(shape_) * data_type_.getSizeInBytes();
    data_ = new std::byte[size_in_bytes];
    is_owned_ = true;
  }

  // Disallow copying: raw pointer with ownership semantics
  CpuStorage(const CpuStorage&) = delete;
  auto operator=(const CpuStorage&) -> CpuStorage& = delete;

  CpuStorage(CpuStorage&& other) noexcept
      : StorageBase(other), data_(other.data_), is_owned_(other.is_owned_) {
    other.data_ = nullptr;
    other.is_owned_ = false;
  }

  auto operator=(CpuStorage&& other) noexcept -> CpuStorage& {
    if (this != &other) {
      if (is_owned_ && data_) {
        delete[] data_;
      }
      this->shape_ = std::move(other.shape_);
      this->strides_ = std::move(other.strides_);
      this->dtype_ = other.dtype_;

      data_ = other.data_;
      is_owned_ = other.is_owned_;
      other.data_ = nullptr;
      other.is_owned_ = false;
    }
    return *this;
  }

  ~CpuStorage() {
    if (is_owned_ && data_) {
      delete[] data_;
    }
  }

  auto isWritable() const -> bool final { return true; }

  auto data() -> std::byte* final { return data_; }

  auto fillWithZeros() -> void final {
    if (data_type() == DataType::Float32) {
      auto* ptr = reinterpret_cast<f32*>(data_);
      std::fill_n(ptr, size(), 0.0f);
    } else if (data_type() == DataType::Float64) {
      auto* ptr = reinterpret_cast<f64*>(data_);
      std::fill_n(ptr, size(), 0.0);
    }
  }

  // Static factories returning type-erased Storage
  static auto create(std::byte* type_erased_ptr, llvm::ArrayRef<i64> shape,
                     DataType data_type, bool is_owned,
                     Layout layout = Layout::RowMajor) -> CpuStorage {
    auto strides = computeStrides(shape, layout);
    return CpuStorage(type_erased_ptr, shape, strides, data_type, is_owned);
  }

  static auto create(std::byte* type_erased_ptr, llvm::ArrayRef<i64> shape,
                     DataType data_type, bool is_owned,
                     llvm::ArrayRef<i64> strides) -> CpuStorage {
    return CpuStorage(type_erased_ptr, shape, strides, data_type, is_owned);
  }

  static auto createUninit(llvm::ArrayRef<i64> shape, DataType data_type,
                           Layout layout = Layout::RowMajor) -> CpuStorage {
    auto strides = computeStrides(shape, layout);
    return CpuStorage(shape, strides, data_type);
  }

  static auto createFilled(Scalar scalar, llvm::ArrayRef<i64> shape,
                           llvm::ArrayRef<i64> strides) -> CpuStorage {
    CpuStorage impl(shape, strides, scalar.data_type());
    auto num_elems = computeNumElems(shape);
    switch (scalar.data_type().kind()) {
      case DataType::Float32: {
        auto* data_ptr = reinterpret_cast<f32*>(impl.data());
        std::fill_n(data_ptr, num_elems, scalar.as<f32>());
        break;
      }
      case DataType::Float64: {
        auto* data_ptr = reinterpret_cast<f64*>(impl.data());
        std::fill_n(data_ptr, num_elems, scalar.as<f64>());
        break;
      }
    }
    return impl;
  }

  static auto createFilled(Scalar value, llvm::ArrayRef<i64> shape,
                           Layout layout = Layout::RowMajor) -> CpuStorage {
    auto strides = computeStrides(shape, layout);
    return createFilled(value, shape, strides);
  }

  static auto createDistributed(llvm::ArrayRef<i64> shape, float mean,
                                float standard_deviation, DataType data_type,
                                Layout layout = Layout::RowMajor)
      -> CpuStorage {
    static thread_local std::mt19937 generator;

    auto num_elems = computeNumElems(shape);
    auto strides = computeStrides(shape, layout);

    switch (data_type.kind()) {
      case DataType::Float32: {
        std::normal_distribution<f32> distribution(mean, standard_deviation);
        CpuStorage impl(shape, strides, data_type);
        auto data_ptr = reinterpret_cast<f32*>(impl.data());
        for (int i = 0; i < num_elems; ++i) {
          data_ptr[i] = distribution(generator);
        }
        return impl;
      }
      case DataType::Float64: {
        std::normal_distribution<f64> distribution(mean, standard_deviation);
        CpuStorage impl(shape, strides, data_type);
        auto data_ptr = reinterpret_cast<f64*>(impl.data());
        for (int i = 0; i < num_elems; ++i) {
          data_ptr[i] = distribution(generator);
        }
        return impl;
      }
    }
    AXON_UNREACHABLE("Unhandled data type");
  }

  static auto createZerosLike(const Storage& storage) -> CpuStorage {
    auto scalar = Scalar(0.0).cast(storage.data_type());
    return createFilled(scalar, storage.shape(), storage.strides());
  }

  static auto fromPythonList(const nb::list& list, DataType data_type)
      -> CpuStorage {
    switch (data_type.kind()) {
      case DataType::Float32: {
        auto shape = computeShape(list);
        auto strides = computeStrides(shape, Layout::RowMajor);
        CpuStorage impl(shape, strides, data_type);
        auto* ptr = reinterpret_cast<f32*>(impl.data());
        auto* fill_ptr = ptr;
        fillBuffer<f32>(list, fill_ptr);
        return impl;
      }
      case DataType::Float64: {
        auto shape = computeShape(list);
        auto strides = computeStrides(shape, Layout::RowMajor);
        CpuStorage impl(shape, strides, data_type);
        auto* ptr = reinterpret_cast<f64*>(impl.data());
        auto* fill_ptr = ptr;
        fillBuffer<f64>(list, fill_ptr);
        return impl;
      }
    }
    AXON_UNREACHABLE("Unhandled data type");
  }

 private:
  std::byte* data_ = nullptr;
  bool is_owned_ = false;
};

template <typename... Args>
class DlPackStorage final : public StorageBase {
 public:
  DlPackStorage(nb::ndarray<Args...> array)
      : StorageBase(DataType::fromDlPack(array.dtype()),
                    llvm::ArrayRef<i64>(array.shape_ptr(), array.ndim()),
                    llvm::ArrayRef<i64>(array.stride_ptr(), array.ndim())),
        ndarray_(std::move(array)) {}

  auto data() -> std::byte* final {
    return reinterpret_cast<std::byte*>(const_cast<void*>(ndarray_.data()));
  }

  auto isWritable() const -> bool final {
    return !llvm::is_one_of<nb::ro, Args...>();
  }

  auto fillWithZeros() -> void final {
    AXON_ASSERT(false, "DlPackStorage is read-only");
  }

 private:
  nb::ndarray<Args...> ndarray_;
};

export using WritableDlPackStorage = DlPackStorage<nb::c_contig>;
export using ReadOnlyDlPackStorage = DlPackStorage<nb::c_contig, nb::ro>;

export class ScalarStorage final : public StorageBase {
 public:
  ScalarStorage(Scalar value, llvm::ArrayRef<i64> shape)
      : StorageBase(value.data_type(), shape,
                    /*strides*/ llvm::ArrayRef<i64>()),
        value_(value) {
    // Set zero strides for broadcasting
    strides_.assign(shape_.size(), 0);
  }

  auto data() -> std::byte* final {
    return const_cast<std::byte*>(value_.bytes().data());
  }

  auto isWritable() const -> bool final { return false; }

  auto fillWithZeros() -> void final {
    switch (data_type().kind()) {
      case DataType::Float32:
        value_ = Scalar(0.0f);
        break;
      case DataType::Float64:
        value_ = Scalar(0.0);
        break;
    }
  }

  static auto create(Scalar value, llvm::ArrayRef<i64> shape) -> Storage {
    return Storage(ScalarStorage(value, shape));
  }

 private:
  Scalar value_;
};

}  // namespace axon
