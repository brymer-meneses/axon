module;

#include <memory>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"

export module axon.python:abi;

import axon.core;
import :tensor;

export namespace axon::abi {

class MemRefDescriptor {
 public:
  static auto create(void* allocated_ptr, void* aligned_ptr, int64_t offset,
                     llvm::ArrayRef<int64_t> shape,
                     llvm::ArrayRef<int64_t> strides) -> MemRefDescriptor* {
    auto rank = shape.size();
    auto shape_size = rank * sizeof(int64_t);
    auto strides_size = rank * sizeof(int64_t);

    auto total_size = sizeof(MemRefDescriptor) + shape_size + strides_size;
    auto* buffer = new std::byte[total_size];
    createInPlace(buffer, allocated_ptr, aligned_ptr, offset, shape, strides);

    return reinterpret_cast<MemRefDescriptor*>(buffer);
  }

  static auto createInPlace(std::byte* buffer, void* allocated_ptr,
                            void* aligned_ptr, int64_t offset,
                            llvm::ArrayRef<int64_t> shape,
                            llvm::ArrayRef<int64_t> strides) -> void {
    auto rank = shape.size();
    auto shape_size = rank * sizeof(int64_t);
    auto strides_size = rank * sizeof(int64_t);

    auto* ptr = reinterpret_cast<MemRefDescriptor*>(buffer);
    *ptr = MemRefDescriptor(allocated_ptr, aligned_ptr, offset);

    auto shape_ptr = buffer + sizeof(MemRefDescriptor);
    auto strides_ptr = shape_ptr + shape_size;

    std::memcpy(shape_ptr, shape.data(), shape_size);
    std::memcpy(strides_ptr, strides.data(), strides_size);
  }

  static auto getAllocSize(int64_t rank) -> size_t {
    auto shape_size = rank * sizeof(int64_t);
    auto strides_size = rank * sizeof(int64_t);
    auto total_size = sizeof(MemRefDescriptor) + shape_size + strides_size;
    return total_size;
  }

 private:
  MemRefDescriptor() = default;
  MemRefDescriptor(void* allocated_ptr, void* aligned_ptr, int64_t offset)
      : allocated_ptr_(allocated_ptr),
        aligned_ptr_(aligned_ptr),
        offset_(offset) {}

 private:
  void* allocated_ptr_;
  void* aligned_ptr_;
  int64_t offset_;
};

struct TensorDescriptor {
  static auto destroy(TensorDescriptor* ptr) -> void {
    auto buffer = reinterpret_cast<std::byte*>(ptr);
    delete[] buffer;
  }

  static auto create(const Tensor& tensor) -> TensorDescriptor* {
    AXON_DCHECK(tensor.hasData(), "Passed tensor is not live");

    auto total_size = MemRefDescriptor::getAllocSize(tensor.rank());
    if (tensor.requiresGrad()) {
      total_size *= 2;
    }

    auto* buffer = new std::byte[total_size];
    auto data_ptr = tensor.data()->data_ptr();
    MemRefDescriptor::createInPlace(buffer, data_ptr, data_ptr, 0,
                                    tensor.data()->shape(),
                                    tensor.data()->strides());

    if (tensor.requiresGrad()) {
      auto grad = tensor.grad();
      auto grad_data_ptr = grad->data()->data_ptr();

      auto* ptr = buffer + MemRefDescriptor::getAllocSize(tensor.rank());
      MemRefDescriptor::createInPlace(ptr, grad_data_ptr, grad_data_ptr, 0,
                                      grad->shape(), grad->data()->strides());
    }

    return reinterpret_cast<TensorDescriptor*>(buffer);
  }
};

}  // namespace axon::abi
