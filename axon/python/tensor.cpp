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

}  // namespace axon
