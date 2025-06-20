module;

#include <memory>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xtensor/containers/xarray.hpp"

export module axon.global_context;

import axon.storage;
import axon.convert;
import axon.ids;
import axon.tensor_metadata;

namespace nb = nanobind;

namespace axon {

export class GlobalContext {
 public:
  GlobalContext() = default;

  static auto Get() -> GlobalContext& {
    static thread_local GlobalContext context;
    return context;
  }

  auto CreateTensor(nb::ndarray<nb::numpy, float> data, bool requires_grad)
      -> TensorId {
    auto data_converted = ConvertNDArrayToXArray<float>(data);
    auto grad_id = data_.Append(xt::zeros_like(data_converted));
    auto data_id = data_.Append(std::move(data_converted));
    return tensor_metadata_.Append(
        TensorMetadata::Create(data_id, grad_id, requires_grad));
  }

  auto Destroy(TensorId id) -> void { tensor_metadata_[id].is_alive = false; }

  auto RequiresGrad(TensorId id) const -> bool {
    return tensor_metadata_[id].requires_grad;
  }

  auto Data(TensorId id) const -> const xt::xarray<float>& {
    return data_[tensor_metadata_[id].data_id];
  }

  auto Gradient(TensorId id) const -> const xt::xarray<float>& {
    return data_[tensor_metadata_[id].grad_id];
  }

 private:
  Storage<TensorId, TensorMetadata> tensor_metadata_;
  Storage<DataId, xt::xarray<float>> data_;
};

}  // namespace axon
