module;

#include <memory>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "xtensor/containers/xarray.hpp"

export module axon.python:tensor;

import axon.base;
import axon.core;

export namespace axon {

struct LazyTensor {
  InstId inst_id;
};

struct Tensor {
  Tensor(Storage& data) : data(std::move(data)), grad(std::nullopt) {}
  Tensor(Storage& data, Storage& grad)
      : data(std::move(data)), grad(std::move(grad)) {}

  auto shape() const -> llvm::ArrayRef<int64_t> { return data.shape(); }
  auto requiresGrad() const -> bool { return grad.has_value(); }

  Storage data;
  std::optional<Storage> grad;
};

class MemRefDescriptor {
 public:
  MemRefDescriptor(const Storage&) {}

  ~MemRefDescriptor() {
    delete[] sizes_;
    delete[] strides_;
  }

 private:
  void* allocated_ptr_;
  void* aligned_ptr_;
  int64_t offset_;
  int64_t* sizes_;
  int64_t* strides_;
};

enum TensorType { RequiresGrad, Normal };

template <TensorType>
struct TensorDescriptor {};

template <>
struct TensorDescriptor<RequiresGrad> {};

template <>
struct TensorDescriptor<Normal> {};

}  // namespace axon
