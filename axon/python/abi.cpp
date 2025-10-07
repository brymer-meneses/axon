module;

#include <memory>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"

export module axon.python:abi;

import axon.base;
import axon.core;

import :tensor;
import :storage;

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

    auto cpu = CpuStorage::create(
        reinterpret_cast<std::byte*>(descriptor->aligned_ptr_), shape,
        data_type,
        /*is_owned=*/false, strides);
    return Storage(std::move(cpu));
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

}  // namespace axon::abi
