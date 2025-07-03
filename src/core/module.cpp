module;

#include <cassert>
#include <flat_map>

#include "llvm/ADT/SmallVector.h"

export module axon.core;

import axon.base.index;
import axon.base.value_store;

export import axon.core.ids;
export import axon.core.inst;

namespace axon {

struct ParameterInfo {
  bool requires_grad;
  llvm::SmallVector<int32_t, 3> shape;
};

struct IntermediaryValue {
  InstId forward_inst_id;
};

struct Dependency {
  InstId tensor_id;
  InstId grad_id;
};

export class Module {
 public:
  auto declare_parameter(ParameterInfo info) -> InstId {
    ParamId param_id = parameters_.emplace(info);
    return forward_insts_.add(insts::DeclareParameter(param_id));
  }

  // Do we need to memoize this?
  auto requires_grad(InstId inst_id) const -> bool {
    llvm::SmallVector<InstId> stack = {inst_id};

    while (not stack.empty()) {
      InstId inst_id = stack.pop_back_val();

      auto& inst = forward_insts_.get(inst_id);
      if (auto param = inst.try_get_as<insts::DeclareParameter>()) {
        ParameterInfo info = parameters_.get(param->param_id);
        return info.requires_grad;
      }

      for (InstId parent_id : inst.parents()) {
        stack.push_back(parent_id);
      }
    }
    return false;
  }

  auto get_intermediary_value(InstId source_id) -> InstId {
    for (auto [value_id, value] : intermediary_values_.iter_values()) {
      if (value.forward_inst_id == source_id) {
        return backward_insts_.add(insts::GetIntermediaryValue(value_id));
      }
    }

    // If we can't find a corresponding IntermediaryValue for this `source_id`,
    // then we create it.
    auto value_id = intermediary_values_.emplace(source_id);
    auto inst_id = backward_insts_.add(insts::GetIntermediaryValue(value_id));
    return inst_id;
  }

  auto accumulate_grad(InstId tensor_id, InstId grad_id) -> void {
    if (gradients_.contains(tensor_id)) {
      InstId prev_grad = gradients_.at(tensor_id);
      grad_id = backward_insts_.add(insts::Add(prev_grad, grad_id));
    }

    gradients_.insert_or_assign(tensor_id, grad_id);
  }

  auto build_backward(InstId tensor_id) {
    auto grad_id = backward_insts_.add(insts::InitialGradient());

    llvm::SmallVector<Dependency> stack = {{tensor_id, grad_id}};
    while (not stack.empty()) {
      auto dep = stack.pop_back_val();
      auto& inst = forward_insts_.get(dep.tensor_id);

      accumulate_grad(dep.tensor_id, dep.grad_id);

      inst.visit([&](const auto op) {
        using InstType = decltype(op);
        if constexpr (InstType::Differentiable) {
          backward(op, dep.grad_id, stack);
        }
      });
    }

    for (auto value_id : intermediary_values_.iter()) {
      auto value = intermediary_values_.get(value_id);
      forward_insts_.emplace(insts::Write(value.forward_inst_id, value_id));
    }
  }

  auto forward_insts() -> ValueStore<InstId, Inst>& { return forward_insts_; }
  auto forward_insts() const -> const ValueStore<InstId, Inst>& {
    return forward_insts_;
  }

  auto backward_insts() -> ValueStore<InstId, Inst>& { return backward_insts_; }
  auto backward_insts() const -> const ValueStore<InstId, Inst>& {
    return backward_insts_;
  }

  auto intermediary_values() const
      -> const ValueStore<IntermediaryValueId, IntermediaryValue>& {
    return intermediary_values_;
  }

  auto intermediary_values()
      -> ValueStore<IntermediaryValueId, IntermediaryValue>& {
    return intermediary_values_;
  }

 private:
  // TODO: explore ways to extract this out of here.
  auto backward(const insts::Add add, const InstId grad_id,
                llvm::SmallVector<Dependency>& deps) -> void {
    if (requires_grad(add.lhs_id)) {
      deps.emplace_back(add.lhs_id, grad_id);
    }
    if (requires_grad(add.rhs_id)) {
      deps.emplace_back(add.rhs_id, grad_id);
    }
  }

  auto backward(const insts::Mul mul, const InstId grad_id,
                llvm::SmallVector<Dependency>& deps) -> void {
    if (requires_grad(mul.lhs_id)) {
      auto rhs_id = get_intermediary_value(mul.rhs_id);
      auto prod_id = backward_insts_.emplace(insts::Mul(rhs_id, grad_id));
      deps.emplace_back(mul.lhs_id, prod_id);
    }
    if (requires_grad(mul.rhs_id)) {
      auto lhs_id = get_intermediary_value(mul.lhs_id);
      auto prod_id = backward_insts_.emplace(insts::Mul(lhs_id, grad_id));
      deps.emplace_back(mul.rhs_id, prod_id);
    }
  }

  auto backward(auto, const InstId, llvm::SmallVector<Dependency>&) -> void {
    static_assert(false,
                  "Operation was declared to be differentiable but has no "
                  "`backward` method.");
  }

 private:
  // Instructions for the backward pass.
  ValueStore<InstId, Inst> backward_insts_;
  // Instructions for the forward pass.
  ValueStore<InstId, Inst> forward_insts_;

  // Parameters that are used by this module.
  ValueStore<ParamId, ParameterInfo> parameters_;

  // Values that are implicitly passed to the backward function.
  ValueStore<IntermediaryValueId, IntermediaryValue> intermediary_values_;

  // Mapping from the inst_id from the forward function to its gradient.
  std::flat_map<InstId, InstId> gradients_;
};

}  // namespace axon
