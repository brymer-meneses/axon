module;

#include "llvm/ADT/ArrayRef.h"
#include "xtensor/containers/xarray.hpp"

export module axon.core:context;

import axon.base;

import :ids;

export namespace axon::core {

class Data {
 public:
  explicit Data(llvm::ArrayRef<int64_t> shape)
      : value_(xt::empty<float>(shape)) {}
  explicit Data(xt::xarray<float> value) : value_(std::move(value)) {}

  // NOTE: We can't directly use `value_.shape()` since xt::xarray<float>
  // internally uses uint64_t for the elements in its shape array. But MLIR
  // tensors expect int64_t.
  auto shape() const -> llvm::SmallVector<int64_t> {
    llvm::SmallVector<int64_t, 4> shape;
    for (auto d : value_.shape()) {
      shape.push_back(static_cast<int64_t>(d));
    }
    return shape;
  }

  auto ref() const -> llvm::ArrayRef<float> {
    return {value_.data(), value_.size()};
  }

 private:
  xt::xarray<float> value_;
};

struct Tensor {
  DataId data_id;
  DataId grad_id;
};

/// Context contains all the information pertaining to the buffers of each
/// tensor.
struct Context {
  // TODO: we should probably intern constant tensors.
  auto create_tensor(xt::xarray<float> value, bool requires_grad) -> Tensor {
    auto data_id = data.emplace(std::move(value));
    auto grad_id = DataId::None;

    if (requires_grad) {
      grad_id = data.emplace(xt::empty_like(value));
    }
    return Tensor(data_id, grad_id);
  }

  ValueStore<DataId, Data> data;
};

}  // namespace axon::core
