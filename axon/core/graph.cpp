module;

#include "axon/base/dcheck.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:graph;

import axon.base;
import std;

import :ids;
import :inst;

namespace axon {

struct Parameter {
  llvm::SmallVector<int64_t> shape;
  bool requires_grad;
  InstId inst_id = InstId::None;
};

struct Data {
  std::vector<float> data;
  llvm::SmallVector<int64_t> shape;

  Data(float value) : data({value}), shape({}) {}
};

export class Graph {
 public:
  auto declareParam(llvm::SmallVector<int64_t> shape, bool requires_grad)
      -> InstId {
    auto input_id = parameters_.emplace(Parameter(shape, requires_grad));
    auto inst_id = insts_.emplace(insts::GetParameter(input_id));
    auto& param = parameters_.get(input_id);
    param.inst_id = inst_id;
    if (requires_grad) {
      gradients_.set(inst_id, InstId::Pending);
    }
    return inst_id;
  }

  auto createConstant(float value) -> InstId {
    auto data_id = data_.emplace(value);
    auto inst_id = insts_.emplace(insts::Constant{data_id});
    inst_to_data_.set(inst_id, data_id);
    return inst_id;
  }

  auto getShape(InstId inst_id) -> llvm::ArrayRef<int64_t> {
    if (auto get_param = insts_.get(inst_id).tryGetAs<insts::GetParameter>()) {
      const auto& param = parameters_.get(get_param->param_id);
      return param.shape;
    }
    return {};
  }

  auto getData(InstId inst_id) -> std::optional<Data*> {
    if (auto data_id = inst_to_data_.get(inst_id)) {
      return &data_.get(data_id);
    }
    return std::nullopt;
  }

  auto checkRequiresGrad(InstId inst_id) const -> bool {
    return gradients_.containsKey(inst_id);
  }

  auto emit(Inst inst) -> InstId { return insts_.emplace(inst); }

  auto createOp(Inst inst) -> InstId {
    auto inst_id = emit(inst);
    auto requires_grad = std::ranges::any_of(
        inst.parents(),
        [this](InstId parent_id) { return checkRequiresGrad(parent_id); });
    if (requires_grad) {
      gradients_.set(inst_id, InstId::Pending);
    }
    return inst_id;
  }

  auto gradients() -> auto& { return gradients_; }
  auto gradients() const -> const auto& { return gradients_; }

  auto insts() -> auto& { return insts_; }
  auto insts() const -> const auto& { return insts_; }

  auto data() -> auto& { return data_; }
  auto data() const -> const auto& { return data_; }

  auto parameters() -> auto& { return parameters_; }
  auto parameters() const -> const auto& { return parameters_; }

 private:
  ValueStore<ParamId, Parameter> parameters_;
  ValueStore<InstId, Inst> insts_;
  IdStore<InstId, DataId> inst_to_data_;
  ValueStore<DataId, Data> data_;

  IdStore<InstId, InstId> gradients_;
};

}  // namespace axon
