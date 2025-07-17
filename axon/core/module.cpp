module;

#include <cassert>
#include <flat_map>
#include <map>
#include <ranges>
#include <string>
#include <variant>

#include "axon/base/dcheck.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "xtensor/containers/xarray.hpp"

export module axon.core:mod;

import axon.base;

import :ids;
import :inst;
import :inst_rules;
import :tensor;

export namespace axon {

class Module;

struct TensorData {
  llvm::SmallVector<int64_t> shape;
};

class Module {
 public:
  auto check_requires_grad(InstId tensor_inst_id) const -> bool {
    return gradients_.contains(tensor_inst_id);
  }
  // Finalizes the module and constructs the backward graph.
  auto create_return(InstId tensor_id) -> void {
    auto grad_id = backward_insts_.add(insts::GetInput(InputId(0)));

    llvm::SmallVector<Dependency> deps = {{tensor_id, grad_id}};
    BackwardBuilder builder{*this, deps};

    while (not deps.empty()) {
      auto dep = deps.pop_back_val();
      auto& inst = forward_insts_.get(dep.tensor_id);

      accumulate_grad(dep.tensor_id, dep.grad_id);

      inst.visit([&](const auto& op) {
        using InstType = std::decay_t<decltype(op)>;
        if constexpr (HasBackward<InstType>) {
          InstHandler<InstType>::backward(op, dep.grad_id, builder);
        }
      });
    }

    for (auto [forward_inst_id, cached_value_inst_id] :
         cached_values_.relations()) {
      auto cached_value = backward_insts_.get(cached_value_inst_id)
                              .get_as<insts::GetCachedValue>();
      forward_insts_.emplace(
          insts::SetCachedValue(forward_inst_id, cached_value.value_id));
    }
  }

  auto emit_inst(Inst inst) -> InstId {
    InstId inst_id = forward_insts_.emplace(inst);
    if (inst.is_expression()) {
      bool requires_grad = llvm::any_of(
          inst.parents(),
          [this](InstId parent_id) { return check_requires_grad(parent_id); });

      gradients_[inst_id] = requires_grad ? InstId::Pending : InstId::Invalid;
    }
    return inst_id;
  }

  auto create_tensor(llvm::SmallVector<int64_t> shape, bool requires_grad)
      -> InstId {
    auto inst_id = forward_insts_.add(insts::LocalTensor{});
    gradients_[inst_id] = requires_grad ? InstId::Pending : InstId::Invalid;
    forward_tensors_[inst_id] = TensorData(shape);
    return inst_id;
  }

  auto declare_input_tensor(llvm::SmallVector<int64_t> shape,
                            bool requires_grad) -> InstId {
    auto grad_inst_id = requires_grad ? InstId::Pending : InstId::Invalid;
    AXON_DCHECK(input_tensors_.size() < std::numeric_limits<int32_t>::max(),
                "Overflow.");

    auto index = static_cast<int32_t>(input_tensors_.size());
    auto inst_id = forward_insts_.add(insts::GetInput(InputId(index)));

    gradients_[inst_id] = grad_inst_id;
    forward_tensors_[inst_id] = TensorData(shape);
    input_tensors_.push_back(inst_id);

    return inst_id;
  }

  auto forward_insts() const -> const auto& { return forward_insts_; }
  auto forward_insts() -> auto& { return forward_insts_; }

  auto backward_insts() const -> const auto& { return backward_insts_; }
  auto backward_insts() -> auto& { return backward_insts_; }

  auto input_tensors() const -> const auto& { return input_tensors_; }
  auto input_tensors() -> auto& { return input_tensors_; }

  auto forward_tensors() const -> const auto& { return forward_tensors_; }
  auto forward_tensors() -> auto& { return forward_tensors_; }

  auto backward_tensors() const -> const auto& { return backward_tensors_; }
  auto backward_tensors() -> auto& { return backward_tensors_; }

  auto cached_values() const -> const auto& { return cached_values_; }
  auto cached_values() -> auto& { return cached_values_; }

 private:
  auto create_backward_inst(Inst inst) -> InstId {
    return backward_insts_.emplace(inst);
  }

  auto get_cached_value(InstId tensor_inst_id) -> InstId {
    auto cached_value_id = cached_values_.get_target(tensor_inst_id);
    if (cached_value_id.has_value()) {
      return cached_value_id;
    }

    auto index = static_cast<int32_t>(cached_values_.size());
    cached_value_id =
        backward_insts_.add(insts::GetCachedValue(CachedValueId(index)));
    cached_values_.create_relation(tensor_inst_id, cached_value_id);
    return cached_value_id;
  }

  auto accumulate_grad(InstId tensor_id, InstId grad_inst_id) -> void {
    AXON_DCHECK(check_requires_grad(tensor_id),
                "Passed `tensor_id` must require gradients");
    InstId current_gradient_id = gradients_.at(tensor_id);
    if (current_gradient_id == InstId::Pending) {
      grad_inst_id =
          create_backward_inst(insts::Add(current_gradient_id, grad_inst_id));
    }

    gradients_[tensor_id] = grad_inst_id;
  }

  // Instructions for the forward pass.
  ValueStore<InstId, Inst> forward_insts_;
  llvm::DenseMap<InstId, TensorData> forward_tensors_;

  // Instructions for the backward pass.
  ValueStore<InstId, Inst> backward_insts_;
  llvm::DenseMap<InstId, TensorData> backward_tensors_;

  llvm::DenseMap<InstId, InstId> gradients_;

  llvm::SmallVector<InstId> input_tensors_;

  RelationalStore<InstId, InstId> cached_values_;

  friend BackwardBuilder;
  friend TensorData;
};

auto BackwardBuilder::get_cached_value(InstId inst_id) -> InstId {
  return module_.get_cached_value(inst_id);
}

auto BackwardBuilder::check_requires_grad(InstId inst_id) const -> bool {
  return module_.check_requires_grad(inst_id);
}

auto BackwardBuilder::backward(InstId inst_id, InstId grad_id) -> void {
  deps_.emplace_back(inst_id, grad_id);
}

auto BackwardBuilder::emit_inst(Inst inst) -> InstId {
  return module_.create_backward_inst(inst);
}

}  // namespace axon
