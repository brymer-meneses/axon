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
  AXON_DCHECK(not module.isFinalized(), "Module must not be finalized.");

  auto& forward_graph = module.forward();
  auto& backward_graph = module.backward();

  auto grad_id = backward_graph.declareInput({}, /*requires_grad=*/false);

  BackwardContext ctx{module};

  llvm::SmallVector<Dependency> work_list;

  auto output_id = forward_graph.output();

  AXON_DCHECK(output_id.isValid(), "`output_id` has no value.");

  work_list.emplace_back(output_id, grad_id);

  while (not work_list.empty()) {
    auto dep = work_list.pop_back_val();

    ctx.accumulateGrad(dep.inst_id, dep.grad_id);

    auto& inst = forward_graph.insts().get(dep.inst_id);
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

  for (InputId input_id : forward_graph.inputs().keys()) {
    auto input_info = forward_graph.inputs().get(input_id);
    auto buffer_id = BufferId(input_id.value());

    if (module.checkRequiresGrad(input_info.inst_id)) {
      backward_graph.emit(insts::AccumulateGrad(buffer_id, input_info.inst_id));
    }
  }

  forward_graph.emit(insts::Return(output_id));
  backward_graph.emit(insts::Return(InstId::None));
}

}  // namespace axon
