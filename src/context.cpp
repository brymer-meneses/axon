module;

#include <memory>
#include <stack>

#include "xtensor/containers/xarray.hpp"

export module axon.context;

import axon.graph;
import axon.tensor;
import axon.executors.eager;

namespace axon {

export class Context {
 public:
  Context()
      : graph_(std::make_shared<Graph>()),
        executor_(std::make_shared<EagerExecutor>(graph_)) {}

  auto create_tensor(xt::xarray<float> data, bool requires_grad) -> Tensor {
    auto inst_id = graph_->create_tensor(data, requires_grad);
    return Tensor(inst_id, graph_);
  }

  auto graph() -> std::shared_ptr<Graph> { return graph_; }
  auto executor() -> std::shared_ptr<EagerExecutor> { return executor_; }

 private:
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<EagerExecutor> executor_;
};

};  // namespace axon
