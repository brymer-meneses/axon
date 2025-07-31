module;

#include "axon/base/dcheck.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "xtensor/containers/xarray.hpp"

export module axon.core:mod;

import axon.base;

import :ids;
import :inst;
import :inst_rules;
import :function;
import :context;

export namespace axon::core {

class Module {
 public:
  Module(Context& context) : context_(context) {}

  auto create_return(InstId tensor_id) -> void {
    InstId grad_id = backward_function.declare_input(/*shape=*/{});

    llvm::SmallVector<Dependency> deps = {{tensor_id, grad_id}};
    BackwardWriter builder{*this, deps};

    while (not deps.empty()) {
      auto dep = deps.pop_back_val();
      auto& inst = forward_function.insts.get(dep.tensor_id);

      accumulate_grad(dep.tensor_id, dep.grad_id);

      inst.visit([&](const auto& op) {
        using InstType = std::decay_t<decltype(op)>;
        if constexpr (HasBackward<InstType>) {
          InstHandler<InstType>::backward(op, dep.grad_id, builder);
        }
      });
    }

    for (auto [forward_inst_id, cached_value_inst_id] :
         cached_values.relations()) {
      auto cached_value = backward_function.insts.get(cached_value_inst_id)
                              .get_as<insts::GetCachedValue>();
      forward_function.emit(
          insts::SetCachedValue(cached_value.cached_value_id, forward_inst_id));
    }

    for (auto input_id : forward_function.inputs.iter()) {
      auto input_inst_id = forward_function.inputs.get(input_id).inst_id;
      if (not forward_function.check_requires_grad(input_inst_id)) {
        continue;
      }

      auto grad_id = gradients.get(input_inst_id);
      backward_function.emit(insts::AccumulateGrad(input_id, grad_id));
    }

    // Register this instruction as a return value.
    forward_function.returns.push_back(tensor_id);
  }

 private:
  auto get_cached_value(InstId tensor_inst_id) -> InstId {
    auto cached_value_id = cached_values.get_target(tensor_inst_id);
    if (cached_value_id.has_value()) {
      return cached_value_id;
    }

    auto index = static_cast<int32_t>(cached_values.size());
    cached_value_id =
        backward_function.emit(insts::GetCachedValue(CachedValueId(index)));
    cached_values.create_relation(tensor_inst_id, cached_value_id);
    return cached_value_id;
  }

  auto accumulate_grad(InstId tensor_id, InstId grad_id) -> void {
    AXON_DCHECK(forward_function.check_requires_grad(tensor_id),
                "Passed `tensor_id` must require gradients");
    InstId current_gradient_id = gradients.get(tensor_id);
    if (current_gradient_id != InstId::Pending) {
      grad_id =
          backward_function.emit(insts::Add(current_gradient_id, grad_id));
    }

    gradients.set(tensor_id, grad_id);
  }

 public:
  ForwardFunction forward_function;
  BackwardFunction backward_function;
  IdStore<InstId, InstId> gradients;

  RelationalStore<InstId, InstId> cached_values;

  friend BackwardWriter;

 private:
  Context& context_;
};

auto BackwardWriter::get_cached_value(InstId inst_id) -> InstId {
  return module_.get_cached_value(inst_id);
}

auto BackwardWriter::check_requires_grad(InstId inst_id) const -> bool {
  return module_.forward_function.check_requires_grad(inst_id);
}

auto BackwardWriter::backward(InstId inst_id, InstId grad_id) -> void {
  deps_.emplace_back(inst_id, grad_id);
}

auto BackwardWriter::emit_inst(Inst inst) -> InstId {
  return module_.backward_function.emit(inst);
}

}  // namespace axon::core
