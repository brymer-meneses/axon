module;

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "xtensor/containers/xarray.hpp"

export module axon.python:tensor;

import axon.base;
import axon.core;

import :storage;

export namespace axon {

struct Tensor {
  Tensor(Storage& data) : data(std::move(data)), grad(std::nullopt) {}
  Tensor(Storage& data, Storage& grad)
      : data(std::move(data)), grad(std::move(grad)) {}

  Storage data;
  std::optional<Storage> grad;
};

}  // namespace axon
