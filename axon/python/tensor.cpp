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

export struct Tensor {
  Tensor(Storage& data) : data(std::move(data)) {}
  Tensor(Storage& data, Storage& grad)
      : data(std::move(data)), grad(std::move(grad)) {}

  Tensor(InstId inst_id) : inst_id(inst_id) {}

  auto isLazy() const -> bool { return inst_id.isValid(); }
  auto hasData() const -> bool { return data.has_value(); }

  auto shape() const -> llvm::ArrayRef<int64_t> {
    if (hasData()) {
      return data->shape();
    }
    std::terminate();
  }
  auto requiresGrad() const -> bool { return grad.has_value(); }

  auto rank() const -> int64_t {
    if (hasData()) {
      return static_cast<int64_t>(data->shape().size());
    }
    std::terminate();
  }

  std::optional<Storage> data;
  std::optional<Storage> grad;
  InstId inst_id = InstId::None;
};

}  // namespace axon
