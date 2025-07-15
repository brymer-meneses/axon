module;

#include <cassert>
#include <string>
#include <variant>

#include "axon/base/dcheck.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:mod;

import axon.base;

import :ids;
import :inst;
import :inst_rules;
import :tensor;

export namespace axon {

class Module {
 public:
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

  auto track_operation(Inst inst) -> InstId {
    auto inst_id = forward_insts_.emplace(inst);
    // If this inst materializes to an expression, then create a corresponding
    // tensor for it.
    if (inst.is_expression()) {
      auto inst_requires_grad = llvm::any_of(
          inst.parents(),
          [this](InstId inst_id) { return requires_grad(inst_id); });
      auto tensor = TensorData::create_local({-1}, inst_requires_grad);
      auto tensor_id = tensors_.emplace(tensor);
      forward_tensors_.create_relation(inst_id, tensor_id);
    }
    return inst_id;
  }

  auto create_constant_tensor(llvm::SmallVector<int64_t> shape,
                              bool requires_grad) -> InstId {
    auto tensor = TensorData::create_local(shape, requires_grad);
    auto tensor_id = tensors_.add(tensor);
    auto inst_id = forward_insts_.add(insts::Constant{});
    forward_tensors_.create_relation(inst_id, tensor_id);
    return inst_id;
  }

  auto declare_input_tensor(Module* module, InstId tensor_inst_id) -> InstId {
    auto tensor = TensorData::create_input(
        module, tensor_inst_id, module->requires_grad(tensor_inst_id));
    auto tensor_id = tensors_.add(tensor);
    auto index = static_cast<int32_t>(input_tensors_.size());
    auto inst_id = forward_insts_.add(insts::GetInput(InputId(index)));
    forward_tensors_.create_relation(inst_id, tensor_id);
    input_tensors_.push_back(tensor_id);
    return inst_id;
  }

  auto get_tensor_data(InstId tensor_inst_id) const -> const TensorData& {
    TensorId tensor_id = forward_tensors_.get_target(tensor_inst_id);
    AXON_DCHECK(tensor_id.has_value(),
                "Passed `tensor_inst_id` is not a tensor");
    return tensors_.get(tensor_id);
  }

  auto input_tensors() const -> const auto& { return input_tensors_; }
  auto input_tensors() -> auto& { return input_tensors_; }

  auto tensors() const -> const auto& { return tensors_; }
  auto tensors() -> auto& { return tensors_; }

  auto forward_insts() const -> const auto& { return forward_insts_; }
  auto forward_insts() -> auto& { return forward_insts_; }

  auto backward_insts() const -> const auto& { return backward_insts_; }
  auto backward_insts() -> auto& { return backward_insts_; }

 private:
  auto create_backward_inst(Inst inst) -> InstId {
    return backward_insts_.emplace(inst);
  }

  auto requires_grad(InstId inst_id) const -> bool {
    // Check if there is a corresponding tensor_id for this inst_id.
    auto tensor_id = forward_tensors_.get_target(inst_id);
    if (tensor_id.has_value()) {
      return tensors_.get(tensor_id).requires_grad();
    }

    return false;
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

  auto accumulate_grad(InstId tensor_inst_id, InstId grad_inst_id) -> void {
    TensorId tensor_id = forward_tensors_.get_target(tensor_inst_id);
    AXON_DCHECK(tensor_id.has_value(), "`tensor_inst_id` must be a tensor.");

    const auto& tensor_data = tensors_.get(tensor_id);
    AXON_DCHECK(tensor_data.requires_grad(),
                "`tensor_inst_id` must be a tensor that requires gradient.");

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

  llvm::SmallVector<TensorId> input_tensors_;

  RelationalStore<InstId, InstId> cached_values_;

  friend BackwardBuilder;
  friend TensorData;
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

auto TensorData::shape() const -> llvm::ArrayRef<int64_t> {
  auto visitor = match{
      [](const LocalTensorData& data) -> llvm::ArrayRef<int64_t> {
        return data.shape;
      },
      [](const InputTensorData& data) -> llvm::ArrayRef<int64_t> {
        const auto& local_data =
            data.module->get_tensor_data(data.tensor_inst_id);
        return local_data.shape();
      },
  };
  return std::visit(visitor, data_);
}

}  // namespace axon
