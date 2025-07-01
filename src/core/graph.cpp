module;

#include <cassert>
#include <flat_map>

#include "llvm/ADT/SmallVector.h"

export module axon.core.graph;

import axon.base.index;
import axon.base.index_map;
import axon.base.value_store;

import axon.core.ids;
import axon.core.inst;

namespace axon {

export struct ParameterInfo {
  bool requires_grad;
};

export class Graph {
 public:
  auto declare_parameter(bool requires_grad) -> InstId {
    ParamId param_id = params_.emplace(requires_grad);
    InstId inst_id = insts_.add(insts::DeclareParameter(param_id));
    return inst_id;
  }

  auto create_inst(Inst inst) -> InstId { return insts_.emplace(inst); }

  // Do we need to memoize this?
  auto requires_grad(InstId inst_id) const -> bool {
    llvm::SmallVector<InstId> stack = {inst_id};

    while (not stack.empty()) {
      InstId inst_id = stack.pop_back_val();

      auto& inst = insts_.get(inst_id);
      if (auto param = inst.try_get_as<insts::DeclareParameter>()) {
        ParameterInfo info = params_.get(param->param_id);
        return info.requires_grad;
      }

      for (InstId parent_id : inst.parents()) {
        stack.push_back(parent_id);
      }
    }
    return false;
  }

  auto add_to_return_values(InstId inst_id) -> void {
    return_values_.emplace_back(inst_id);
  }

  auto insts() -> ValueStore<InstId, Inst>& { return insts_; }
  auto insts() const -> const ValueStore<InstId, Inst>& { return insts_; }

 private:
  ValueStore<InstId, Inst> insts_;
  ValueStore<ParamId, ParameterInfo> params_;
  llvm::SmallVector<InstId> return_values_;
};

export class BackwardFunction final : public Graph {
 public:
  using Graph::Graph;

  // To reuse the values computed in the forward graph that is needed by the
  // backward graph, we pass the required "intermediary_values" in the backward
  // function.
  auto get_or_declare_intermediary_value(InstId source_id) -> InstId {
    if (not intermediary_values_.contains(source_id)) {
      auto localized = declare_parameter(/*requires_grad=*/false);
      intermediary_values_.insert({source_id, localized});
      return localized;
    }
    return intermediary_values_.find(source_id)->second;
  }

  auto accumulate_grad(InstId tensor_id, InstId grad_id) -> void {
    if (gradients_.contains(tensor_id)) {
      InstId prev_grad = gradients_.at(tensor_id);
      grad_id = create_inst(insts::Add(prev_grad, grad_id));
    }

    gradients_.insert_or_assign(tensor_id, grad_id);
  }

  auto intermediary_values() const -> const std::flat_map<InstId, InstId>& {
    return intermediary_values_;
  }

  auto intermediary_values() -> std::flat_map<InstId, InstId>& {
    return intermediary_values_;
  }

 private:
  // Mapping from the inst_id from the forward function to the backward
  // function. The `intermediary_values_` are parameters that are passed by the
  // forward function which are needed by the backward function to avoid
  // recomputation.
  std::flat_map<InstId, InstId> intermediary_values_;

  // Mapping from the inst_id from the forward function to its gradient.
  std::flat_map<InstId, InstId> gradients_;
};

export class ForwardFunction final : public Graph {
  struct Dependency {
    // Lives in the parent graph
    InstId tensor_id;
    // local gradient lives here.
    InstId grad_id;
  };

 public:
  auto build_backward(InstId tensor_id) -> BackwardFunction {
    BackwardFunction backward_function;
    InstId grad_id =
        backward_function.declare_parameter(/*requires_grad=*/true);

    llvm::SmallVector<Dependency> stack = {{tensor_id, grad_id}};
    while (not stack.empty()) {
      auto dep = stack.pop_back_val();

      Inst& inst = insts().get(dep.tensor_id);
      backward_function.accumulate_grad(dep.tensor_id, dep.grad_id);

      inst.visit(match{
          [&](const insts::Add add) {
            if (requires_grad(add.lhs_id)) {
              stack.emplace_back(add.lhs_id, dep.grad_id);
            }
            if (requires_grad(add.rhs_id)) {
              stack.emplace_back(add.rhs_id, dep.grad_id);
            }
          },
          [&](const insts::Mul mul) {
            if (requires_grad(mul.lhs_id)) {
              InstId rhs_id =
                  backward_function.get_or_declare_intermediary_value(
                      mul.rhs_id);
              InstId prod_id = backward_function.create_inst(
                  insts::Mul(rhs_id, dep.grad_id));
              stack.emplace_back(mul.lhs_id, prod_id);
            }
            if (requires_grad(mul.rhs_id)) {
              InstId lhs_id =
                  backward_function.get_or_declare_intermediary_value(
                      mul.lhs_id);
              InstId prod_id = backward_function.create_inst(
                  insts::Mul(lhs_id, dep.grad_id));
              stack.emplace_back(mul.rhs_id, prod_id);
            }
          },
          [](const auto _) {},
      });
    }

    return backward_function;
  }
};

}  // namespace axon
