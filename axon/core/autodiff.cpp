module;

#include "axon/base/dcheck.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:autodiff;

import :inst_rules;
import :graph;
import :ids;
import :mod;

export namespace axon {

auto finalize(Module& module) -> void {
  AXON_DCHECK(not module.is_finalized(), "Module must not be finalized.");

  auto grad_id = module.backward().declare_input({}, /*requires_grad=*/false);

  BackwardContext ctx{module};

  llvm::SmallVector<Dependency> work_list;
  auto& forward_insts = module.forward().insts();
  auto output_id = forward_insts.last();
  AXON_DCHECK(output_id.has_value(), "`output_id` has no value.");

  work_list.emplace_back(output_id, grad_id);

  while (not work_list.empty()) {
    auto dep = work_list.pop_back_val();

    ctx.accumulate_grad(dep.inst_id, dep.grad_id);

    auto& inst = forward_insts.get(dep.inst_id);
    inst.visit([&](const auto& op) {
      using InstType = std::decay_t<decltype(op)>;
      if constexpr (HasBackwardRule<InstType>) {
        auto new_deps = BackwardRule<InstType>::apply(op, grad_id, ctx);
        for (auto dep : new_deps) {
          work_list.emplace_back(dep);
        }
      }
    });
  }

  for (InputId input_id : module.forward().inputs().iter()) {
    auto input_info = module.forward().inputs().get(input_id);
    auto buffer_id = BufferId(input_id.value());
    module.backward().emit(
        insts::AccumulateGrad(buffer_id, input_info.inst_id));
  }

  module.forward().emit(insts::Return(output_id));
  module.backward().emit(insts::Return(InstId::None));
}

}  // namespace axon
