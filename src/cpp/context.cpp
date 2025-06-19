module;

#include <memory>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xtensor/containers/xarray.hpp"

export module axon.context;

import axon.common;
import axon.convert;

namespace nb = nanobind;

namespace axon {

export struct TensorId : IndexBase {
  using IndexBase::IndexBase;

  static const TensorId Invalid;
};

inline constexpr TensorId TensorId::Invalid = TensorId(-1);

export class TensorMetadata {
 public:
  TensorMetadata(xt::xarray<float> data, bool requires_grad)
      : data(data), grad(xt::zeros_like(data)), requires_grad(requires_grad) {}

  xt::xarray<float> data;
  xt::xarray<float> grad;
  bool requires_grad = false;
};

export class Context {
 public:
  Context() = default;

  static auto GetCurrent() -> std::shared_ptr<Context> {
    return s_current_context;
  }

  static auto SetCurrent(std::shared_ptr<Context> context) -> void {
    s_current_context = context;
  }

  auto CreateTensor(const nb::ndarray<nb::numpy, float>& data,
                    bool requires_grad) -> TensorId {
    auto data_converted = ConvertNDArrayToXArray<float>(data);

    return tensor_metadata_.Append(
        TensorMetadata(data_converted, requires_grad));
  }

  auto GetData(TensorId id) const -> const xt::xarray<float>& {
    return tensor_metadata_[id].data;
  }

 private:
  static thread_local std::shared_ptr<Context> s_current_context;

  BasicStorage<TensorId, TensorMetadata> tensor_metadata_;
};

thread_local std::shared_ptr<Context> Context::s_current_context = nullptr;

}  // namespace axon
