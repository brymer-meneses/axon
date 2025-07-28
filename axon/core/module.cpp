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

export namespace axon {

class Module;

class Data {
 public:
  explicit Data(llvm::ArrayRef<int64_t> shape)
      : value_(xt::empty<float>(shape)) {}
  explicit Data(xt::xarray<float> value) : value_(std::move(value)) {}

  // We can't use `value_.shape()` since xt::xarray<float> internally uses
  // uint64_t for the elements in its shape array. But MLIR tensors expect
  // int64_t :<
  auto shape() const -> llvm::SmallVector<int64_t> {
    llvm::SmallVector<int64_t, 4> shape;
    for (auto d : value_.shape()) {
      shape.push_back(d);
    }
    return shape;
  }

  auto ref() const -> llvm::ArrayRef<float> {
    return {value_.data(), value_.size()};
  }

 private:
  xt::xarray<float> value_;
};

class Module {
 public:
  auto check_requires_grad(InstId tensor_id) const -> bool {
    InstId grad_id = gradients_.get(tensor_id);
    return grad_id.is_pending() or grad_id.has_value();
  }
  // Finalizes the module and constructs the backward graph.
  auto create_return(InstId tensor_id) -> void {
    auto grad_id = backward_insts_.add(insts::InitialGradient{});

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
          insts::SetCachedValue(cached_value.cached_value_id, forward_inst_id));
    }

    for (auto [i, input_inst_id] : llvm::enumerate(input_tensors_)) {
      if (not check_requires_grad(input_inst_id)) {
        continue;
      }
      auto grad_id = gradients_.get(input_inst_id);
      backward_insts_.emplace(insts::AccumulateGrad(InputId(i), grad_id));
    }
  }

  auto emit_inst(Inst inst) -> InstId {
    InstId inst_id = forward_insts_.emplace(inst);
    if (inst.is_expression()) {
      bool requires_grad = llvm::any_of(
          inst.parents(),
          [this](InstId parent_id) { return check_requires_grad(parent_id); });

      gradients_.set(inst_id, requires_grad ? InstId::Pending : InstId::None);
    }
    return inst_id;
  }

  auto create_tensor(xt::xarray<float> value, bool requires_grad) -> InstId {
    auto inst_id = forward_insts_.add(insts::LocalTensor{});
    auto data_id = data_.emplace(std::move(value));

    gradients_.set(inst_id, requires_grad ? InstId::Pending : InstId::None);
    forward_data_.set(inst_id, data_id);

    return inst_id;
  }

  auto declare_input_tensor(llvm::SmallVector<int64_t> shape,
                            bool requires_grad) -> InstId {
    AXON_DCHECK(input_tensors_.size() < std::numeric_limits<int32_t>::max(),
                "Overflow.");
    auto index = static_cast<int32_t>(input_tensors_.size());
    auto inst_id = forward_insts_.add(insts::GetInput(InputId(index)));
    auto data_id = data_.emplace(shape);

    gradients_.set(inst_id, requires_grad ? InstId::Pending : InstId::None);
    forward_data_.set(inst_id, data_id);
    input_tensors_.push_back(inst_id);

    return inst_id;
  }

  auto get_data(InstId inst_id, bool is_forward) -> Data& {
    auto data_id =
        is_forward ? forward_data_.get(inst_id) : backward_data_.get(inst_id);
    return data_.get(data_id);
  }

  auto get_inst(InstId inst_id, bool is_forward) -> const Inst& {
    return is_forward ? forward_insts_.get(inst_id)
                      : backward_insts_.get(inst_id);
  }

  auto forward_insts() const -> const auto& { return forward_insts_; }
  auto backward_insts() const -> const auto& { return backward_insts_; }
  auto input_tensors() const -> const auto& { return input_tensors_; }
  auto cached_values() const -> const auto& { return cached_values_; }

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
    InstId current_gradient_id = gradients_.get(tensor_id);
    if (current_gradient_id != InstId::Pending) {
      grad_inst_id =
          create_backward_inst(insts::Add(current_gradient_id, grad_inst_id));
    }

    gradients_.set(tensor_id, grad_inst_id);
  }

  // Instructions for the forward pass.
  ValueStore<InstId, Inst> forward_insts_;
  IdStore<InstId, DataId> forward_data_;

  // Instructions for the backward pass.
  ValueStore<InstId, Inst> backward_insts_;
  IdStore<InstId, DataId> backward_data_;

  IdStore<InstId, InstId> gradients_;
  ValueStore<DataId, Data> data_;

  llvm::SmallVector<InstId> input_tensors_;

  RelationalStore<InstId, InstId> cached_values_;

  friend BackwardBuilder;
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
