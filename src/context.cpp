module;

#include <memory>

#include "xtensor/containers/xarray.hpp"

export module axon.context;

import axon.graph;
import axon.tensor;

namespace axon {

export class Context {
 public:
  Context() : graph_(std::make_shared<Graph>()) {}

  auto create_tensor(xt::xarray<float> data, bool requires_grad) -> Tensor {
    auto inst_id = graph_->create_tensor(data, requires_grad);
    return Tensor(inst_id, graph_);
  }

  auto graph() -> std::shared_ptr<Graph> { return graph_; }

 private:
  std::shared_ptr<Graph> graph_;
};

};  // namespace axon
