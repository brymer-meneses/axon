module;

#include <memory>
#include <optional>
#include <ranges>

#include "axon/base/dcheck.h"
#include "llvm/ADT/SmallVector.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xtensor/containers/xarray.hpp"

export module axon.python:tensor;

import axon.base;
import axon.core;
import axon.mlir;

namespace nb = nanobind;

namespace axon {

export struct LazyTensor {
  InstId inst_id;
};

export struct Tensor {
  Tensor(Storage& data) : data(std::move(data)), grad(std::nullopt) {}
  Tensor(Storage& data, Storage& grad)
      : data(std::move(data)), grad(std::move(grad)) {}

  auto shape() const -> llvm::ArrayRef<int64_t> { return data.shape(); }
  auto requiresGrad() const -> bool { return grad.has_value(); }
  auto rank() const -> int64_t {
    return static_cast<int64_t>(data.shape().size());
  }

  Storage data;
  std::optional<Storage> grad;
};

}  // namespace axon
