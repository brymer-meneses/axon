module;

#include <cassert>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:mod;

import axon.base;

import :ids;
import :inst;
import :inst_rules;

export namespace axon {

struct TensorData {
  auto requires_grad() const -> bool {
    return grad_inst_id.has_value() or grad_inst_id == InstId::Pending;
  }

  InstId grad_inst_id;
  std::string name = "unnamed";
  llvm::SmallVector<int64_t> shape = {};
  // TODO: this should optionally contain a reference to the Module when it is
  // a foreign tensor.
};

class Module {
 public:
  auto finalize(InstId tensor_id) -> void {
    auto grad_id =
        backward_insts_.add(insts::GetFunctionArgument(ArgumentId(0)));

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
                              .get_as_unchecked<insts::GetCachedValue>();
      forward_insts_.emplace(
          insts::SetCachedValue(forward_inst_id, cached_value.value_id));
    }
  }

 private:
  auto create_forward_inst(Inst inst) -> InstId {
    auto inst_id = forward_insts_.emplace(inst);
    // If this inst materializes to an expression, then create a corresponding
    // tensor for it.
    if (inst.is_expression()) {
      auto inst_requires_grad = llvm::any_of(
          inst.parents(),
          [this](InstId inst_id) { return requires_grad(inst_id); });
      TensorData tensor{.grad_inst_id = inst_requires_grad ? InstId::Pending
                                                           : InstId::Invalid};
      auto tensor_id = tensors_.emplace(tensor);
      forward_tensors_.create_relation(inst_id, tensor_id);
    }

    return inst_id;
  }

  auto create_backward_inst(Inst inst) -> InstId {
    auto inst_id = backward_insts_.emplace(inst);

    // If this inst materializes to an expression, then create a corresponding
    // tensor for it.
    if (inst.is_expression()) {
      auto inst_requires_grad = llvm::any_of(
          inst.parents(),
          [this](InstId inst_id) { return requires_grad(inst_id); });
      TensorData tensor{.grad_inst_id = inst_requires_grad ? InstId::Pending
                                                           : InstId::Invalid};
      auto tensor_id = tensors_.emplace(tensor);
      backward_tensors_.create_relation(inst_id, tensor_id);
    }

    return inst_id;
  }

  auto requires_grad(InstId inst_id) const -> bool {
    // Check if there is a corresponding tensor_id for this inst_id.
    auto tensor_id = forward_tensors_.get_right(inst_id);
    if (tensor_id.has_value()) {
      return tensors_.get(tensor_id).requires_grad();
    }

    return false;
  }

  auto get_cached_value(InstId tensor_inst_id) -> InstId {
    auto cached_value_id = cached_values_.get_right(tensor_inst_id);
    if (cached_value_id.has_value()) {
      return cached_value_id;
    }

    auto index = static_cast<int32_t>(cached_values_.size());
    cached_value_id =
        backward_insts_.add(insts::GetCachedValue(CachedValueId(index)));
    cached_values_.create_relation(tensor_inst_id, cached_value_id);
    return cached_value_id;
  }

  auto accumulate_grad(InstId tensor_inst_id, InstId grad_inst_id) -> void {
    TensorId tensor_id = forward_tensors_.get_right(tensor_inst_id);
    assert(tensor_id.has_value());

    const auto& tensor_data = tensors_.get(tensor_id);
    assert(tensor_data.grad_inst_id.has_value() or
           tensor_data.grad_inst_id == InstId::Pending);

    if (tensor_data.grad_inst_id.has_value()) {
      auto inst = insts::Add(tensor_data.grad_inst_id, grad_inst_id);
      grad_inst_id = create_backward_inst(inst);
    }

    // We can't make `tensor_data` a mutable ref since `create_inst` can
    // invalidate `tensor_data`. Maybe explore creating a `Handle<T>` CRTP
    // to do this automatically for us?
    tensors_.get(tensor_id).grad_inst_id = grad_inst_id;
  }

  // Instructions for the forward pass.
  ValueStore<InstId, Inst> forward_insts_;
  RelationalStore<InstId, TensorId> forward_tensors_;

  // Instructions for the backward pass.
  ValueStore<InstId, Inst> backward_insts_;
  RelationalStore<InstId, TensorId> backward_tensors_;

  ValueStore<TensorId, TensorData> tensors_;

  llvm::SmallVector<TensorId> foreign_tensors_;

  RelationalStore<InstId, InstId> cached_values_;

  friend BackwardBuilder;
};

auto BackwardBuilder::get_cached_value(InstId inst_id) -> InstId {
  return module_.get_cached_value(inst_id);
}

auto BackwardBuilder::check_requires_grad(InstId inst_id) const -> bool {
  return module_.requires_grad(inst_id);
}

auto BackwardBuilder::backward(InstId inst_id, InstId grad_id) -> void {
  deps_.emplace_back(inst_id, grad_id);
}

auto BackwardBuilder::emit_inst(Inst inst) -> InstId {
  return module_.create_backward_inst(inst);
}

}  // namespace axon
