module;

#include "axon/base/dcheck.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:graph;

import axon.base;
import std;

import :ids;
import :inst;
import :context;

namespace axon {

struct Input {
  llvm::SmallVector<int64_t> shape;
  bool requires_grad;
  InstId inst_id = InstId::None;
};

export class Graph {
 public:
  auto declare_input(llvm::SmallVector<int64_t> shape, bool requires_grad)
      -> InstId {
    auto input_id = inputs_.emplace(Input(shape, requires_grad));
    auto inst_id = insts_.emplace(insts::GetInput(input_id));
    auto& input = inputs_.get(input_id);
    input.inst_id = inst_id;
    return inst_id;
  }

  // This method is called by the backward graph to generate an inst that
  // corresponds to the `inst_id` in the forward graph.
  auto get_cached_value(InstId inst_id) -> insts::GetCachedValue {
    // If there is an existing cached_value_id for this inst_id then just return
    // that instead.
    if (auto existing_id = cached_values_.get(inst_id);
        existing_id.has_value()) {
      return insts::GetCachedValue(existing_id);
    }
    // Otherwise create a new cached_value_id, for this inst.
    auto cached_value_index = static_cast<int32_t>(cached_values_.size());
    auto cached_value_id = CachedValueId(cached_value_index);
    // Create an instruction that sets the value of this cached_value.
    insts_.emplace(insts::SetCachedValue(cached_value_id, inst_id));
    cached_values_.set(inst_id, cached_value_id);
    return insts::GetCachedValue(cached_value_id);
  }

  auto get_shape(InstId inst_id) -> llvm::ArrayRef<int64_t> {
    if (auto get_input = insts_.get(inst_id).try_get_as<insts::GetInput>()) {
      auto& input_info = inputs_.get(get_input->input_id);
      return input_info.shape;
    }
    return {};
  }

  auto emit(Inst inst) -> InstId {
    auto inst_id = insts_.emplace(inst);
    // If this is instruction corresponds to an expression, then we create a
    // slot for it's data.
    if (inst.is_expression()) {
      data_.set(inst_id, DataId::Pending);
    }
    return inst_id;
  }

  auto set_output(InstId output) -> void {
    AXON_DCHECK(not output_.has_value(), "Set output called twice.");
    output_ = output;
  }

  auto insts() -> auto& { return insts_; }
  auto insts() const -> const auto& { return insts_; }

  auto inputs() -> auto& { return inputs_; }
  auto inputs() const -> const auto& { return inputs_; }

  auto data() -> auto& { return data_; }
  auto data() const -> const auto& { return data_; }

  auto output() const -> InstId { return output_; }

  auto cached_values() -> auto& { return cached_values_; }
  auto cached_values() const -> const auto& { return cached_values_; }

 private:
  ValueStore<InputId, Input> inputs_;
  ValueStore<InstId, Inst> insts_;

  IdStore<InstId, DataId> data_;
  IdStore<InstId, CachedValueId> cached_values_;

  InstId output_ = InstId::None;
};

}  // namespace axon
