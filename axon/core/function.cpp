module;

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:function;

import :ids;
import :inst;

export namespace axon::core {

struct Input {
  llvm::SmallVector<int64_t> shape;
  bool requires_grad;
  InstId inst_id;
};

struct FunctionBase {
  ValueStore<InstId, Inst> insts;
  IdStore<InstId, DataId> data;
  ValueStore<InputId, Input> inputs;
};

struct ForwardFunction : public FunctionBase {
  auto check_requires_grad(InstId target_inst_id) const -> bool {
    for (auto inst_id : differentiables) {
      if (inst_id == target_inst_id) {
        return true;
      }
    }
    return false;
  }

  auto emit(Inst inst) -> InstId {
    InstId inst_id = insts.emplace(inst);
    if (not inst.is_expression()) {
      return inst_id;
    }

    bool requires_grad = llvm::any_of(inst.parents(), [this](InstId parent_id) {
      return check_requires_grad(parent_id);
    });
    if (requires_grad) {
      differentiables.push_back(inst_id);
    }
    return inst_id;
  }

  auto declare_input(llvm::SmallVector<int64_t> shape, bool requires_grad)
      -> InstId {
    auto input_id = inputs.add(Input(shape, requires_grad, InstId::None));
    auto inst_id = insts.add(insts::GetInput(input_id));
    inputs.get(input_id).inst_id = inst_id;
    if (requires_grad) {
      differentiables.push_back(inst_id);
    }
    return inst_id;
  }

  llvm::SmallVector<InstId, 1> returns;
  llvm::SmallVector<InstId> differentiables;
};

struct BackwardFunction : public FunctionBase {
  auto emit(Inst inst) -> InstId {
    InstId inst_id = insts.emplace(inst);
    return inst_id;
  }

  auto declare_input(llvm::SmallVector<int64_t> shape) -> InstId {
    auto input_id = inputs.add(Input(shape, false, InstId::None));
    auto inst_id = insts.add(insts::GetInput(input_id));
    inputs.get(input_id).inst_id = inst_id;
    return inst_id;
  }
};

}  // namespace axon::core
