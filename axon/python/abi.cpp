module;

#include <memory>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"

export module axon.python:abi;

import axon.base;
import axon.core;

import :tensor;

namespace axon::abi {

export class MemRefDescriptor {
 public:
  static auto destroy(MemRefDescriptor* ptr) -> void {
    auto buffer = reinterpret_cast<std::byte*>(ptr);
    delete[] buffer;
  }

  static auto createEmpty(u64 rank) -> MemRefDescriptor* {
    auto size = getAllocSize(rank);
    auto* buffer = new std::byte[size];
    return reinterpret_cast<MemRefDescriptor*>(buffer);
  }

  static auto createStorage(MemRefDescriptor* descriptor, DataType data_type,
                            u64 rank) -> Storage {
    auto shape_size = rank * sizeof(i64);
    auto ptr = reinterpret_cast<std::byte*>(descriptor);

    auto shape_ptr = reinterpret_cast<i64*>(ptr + sizeof(MemRefDescriptor));
    auto strides_ptr =
        reinterpret_cast<i64*>(ptr + sizeof(MemRefDescriptor) + shape_size);

    llvm::ArrayRef<i64> shape(shape_ptr, rank);
    llvm::ArrayRef<i64> strides(strides_ptr, rank);

    AXON_DCHECK(descriptor->offset_ == 0);

    return Storage::create(
        reinterpret_cast<std::byte*>(descriptor->aligned_ptr_), shape,
        data_type,
        // TODO: Storage should take a function ptr to delete the data
        /*is_owned=*/false, strides);
  }

  static auto create(void* allocated_ptr, void* aligned_ptr, i64 offset,
                     llvm::ArrayRef<i64> shape, llvm::ArrayRef<i64> strides)
      -> MemRefDescriptor* {
    auto rank = shape.size();
    auto shape_size = rank * sizeof(i64);
    auto strides_size = rank * sizeof(i64);

    auto total_size = sizeof(MemRefDescriptor) + shape_size + strides_size;
    auto* buffer = new std::byte[total_size];
    createInPlace(buffer, allocated_ptr, aligned_ptr, offset, shape, strides);

    return reinterpret_cast<MemRefDescriptor*>(buffer);
  }

  static auto createInPlace(std::byte* buffer, void* allocated_ptr,
                            void* aligned_ptr, i64 offset,
                            llvm::ArrayRef<i64> shape,
                            llvm::ArrayRef<i64> strides) -> void {
    auto rank = shape.size();
    auto shape_size = rank * sizeof(i64);
    auto strides_size = rank * sizeof(i64);

    auto* ptr = reinterpret_cast<MemRefDescriptor*>(buffer);
    *ptr = MemRefDescriptor(allocated_ptr, aligned_ptr, offset);

    auto shape_ptr = buffer + sizeof(MemRefDescriptor);
    auto strides_ptr = shape_ptr + shape_size;

    std::memcpy(shape_ptr, shape.data(), shape_size);
    std::memcpy(strides_ptr, strides.data(), strides_size);
  }

  static auto getAllocSize(u64 rank) -> u64 {
    auto shape_size = rank * sizeof(i64);
    auto strides_size = rank * sizeof(i64);
    auto total_size = sizeof(MemRefDescriptor) + shape_size + strides_size;
    return total_size;
  }

 private:
  MemRefDescriptor() = default;
  MemRefDescriptor(void* allocated_ptr, void* aligned_ptr, i64 offset)
      : allocated_ptr_(allocated_ptr),
        aligned_ptr_(aligned_ptr),
        offset_(offset) {}

 private:
  void* allocated_ptr_;
  void* aligned_ptr_;
  i64 offset_;
};

export struct TensorDescriptor {
  static auto destroy(TensorDescriptor* ptr) -> void {
    auto buffer = reinterpret_cast<std::byte*>(ptr);
    delete[] buffer;
  }

  static auto create(const Tensor& tensor) -> TensorDescriptor* {
    AXON_DCHECK(tensor.isEvaluated());

    auto total_size = MemRefDescriptor::getAllocSize(tensor.rank());
    if (tensor.requiresGrad()) {
      total_size *= 2;
    }

    auto* buffer = new std::byte[total_size];
    auto descriptor = reinterpret_cast<TensorDescriptor*>(buffer);

    setStorage(descriptor, tensor);

    return descriptor;
  }

  static auto setStorage(TensorDescriptor* descriptor, const Tensor& tensor)
      -> void {
    auto buffer = reinterpret_cast<std::byte*>(descriptor);

    AXON_DCHECK(tensor.storage() != nullptr);

    auto data_ptr = tensor.storage()->data_ptr();

    AXON_DCHECK(data_ptr != nullptr);

    MemRefDescriptor::createInPlace(buffer, data_ptr, data_ptr, 0,
                                    tensor.storage()->shape(),
                                    tensor.storage()->strides());

    if (tensor.requiresGrad()) {
      auto grad = tensor.grad();
      AXON_DCHECK(grad != nullptr);

      auto grad_data_ptr = grad->storage()->data_ptr();

      auto* ptr = buffer + MemRefDescriptor::getAllocSize(tensor.rank());
      MemRefDescriptor::createInPlace(ptr, grad_data_ptr, grad_data_ptr, 0,
                                      grad->shape(),
                                      grad->storage()->strides());
    }
  }
};

}  // namespace axon::abi
