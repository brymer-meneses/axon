module;

#include <cassert>
#include <vector>

export module axon.tensor;

import axon.common;

namespace axon {

export struct Tensor {
  bool requires_grad = false;
  bool is_optimizable = false;

  std::vector<int16_t> shape;
};

export struct TensorId : IdBase<TensorId, Tensor> {
  using ValueType = Tensor;
  using IdBase::IdBase;
};

}  // namespace axon
