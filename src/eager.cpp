module;

#include <memory>
#include <optional>
#include <ranges>
#include <variant>

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/containers/xarray.hpp"

export module axon.executors.eager;

import axon.graph;
import axon.inst;
import axon.ids;
import axon.storage;

namespace axon {

template <typename... Args>
struct match : Args... {
  using Args::operator()...;
};

export class EagerExecutor final : GraphExecutor {
 public:
  EagerExecutor(std::shared_ptr<Graph> graph) : graph_(graph) {}

  auto forward() -> void {
    auto to_drop =
        last_executed_id_.has_value() ? last_executed_id_.value() : 0;

    for (auto inst_id : graph_->insts().iter() | std::views::drop(to_drop)) {
      last_executed_id_ = inst_id;

      auto visitor = match{
          [&](const Create& op) {
            if (graph_->requires_grad(inst_id)) {
              zero_grad(inst_id);
            }
          },
          [&](const MatMul& op) {
            const auto& lhs = graph_->insts().get(op.lhs_id);
            const auto& rhs = graph_->insts().get(op.rhs_id);

            const auto& lhs_data = *graph_->data().get(lhs.data_id).storage;
            const auto& rhs_data = *graph_->data().get(rhs.data_id).storage;

            const auto result = xt::linalg::dot(lhs_data, rhs_data);

            auto& inst = graph_->insts().get(inst_id);
            inst.data_id = graph_->create_data(result);
          },
          [&](const Add& op) {
            const auto& lhs = graph_->insts().get(op.lhs_id);
            const auto& rhs = graph_->insts().get(op.rhs_id);

            const auto& lhs_data = *graph_->data().get(lhs.data_id).storage;
            const auto& rhs_data = *graph_->data().get(rhs.data_id).storage;

            const auto result = lhs_data + rhs_data;

            auto& inst = graph_->insts().get(inst_id);
            inst.data_id = graph_->create_data(result);
          },
          [&](const Mul& op) {
            const auto& lhs = graph_->insts().get(op.lhs_id);
            const auto& rhs = graph_->insts().get(op.rhs_id);

            const auto& lhs_data = *graph_->data().get(lhs.data_id).storage;
            const auto& rhs_data = *graph_->data().get(rhs.data_id).storage;

            const auto result = lhs_data * rhs_data;

            auto& inst = graph_->insts().get(inst_id);
            inst.data_id = graph_->create_data(result);
          },
      };

      std::visit(visitor, graph_->insts().get(inst_id).op);
    }
  }

  auto backward(InstId inst_id) -> void {}

  auto zero_grad(InstId inst_id) -> void {
    if (not graph_->requires_grad(inst_id)) {
      throw std::runtime_error(
          "Cannot zero the gradient of a tensor which does not require a "
          "gradient");
    }

    auto& inst = graph_->insts().get(inst_id);
    const auto& data = graph_->data().get(inst.data_id);

    // This is a new tensor that has been just allocated.
    if (inst.grad_id == DataId::Pending) {
      inst.grad_id = graph_->create_data(xt::zeros_like(*data.storage));
    }
  }

 private:
  InstId last_executed_id_ = InstId::Invalid;
  std::shared_ptr<Graph> graph_;
};

}  // namespace axon
