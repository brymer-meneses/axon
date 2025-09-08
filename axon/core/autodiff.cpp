module;

#include "axon/base/dcheck.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:autodiff;

import :backward_rules;
import :graph;
import :ids;

export namespace axon {

auto backward(Graph& graph, InstId output_id, InstId grad_id = InstId::None)
    -> void {
  AXON_DCHECK(output_id.isValid(), "`output_id` has no value.");

  if (not grad_id.isValid()) {
    grad_id = graph.createOp(insts::OnesLike(output_id));
  }

  llvm::SmallVector<Dependency> work_list;
  work_list.emplace_back(output_id, grad_id);

  BackwardContext ctx{graph};

  while (not work_list.empty()) {
    auto dep = work_list.pop_back_val();

    ctx.accumulateGrad(dep);

    // This should take by copy since `graph.insts` will grow.
    auto inst = graph.insts().get(dep.inst_id);

    inst.visit([&, dep](const auto& op) {
      using InstType = std::decay_t<decltype(op)>;
      if constexpr (HasBackwardRule<InstType>) {
        auto new_deps = BackwardRule<InstType>::apply(op, dep.grad_id, ctx);
        for (auto dep : new_deps) {
          work_list.emplace_back(dep);
        }
      }
    });
  }

  for (auto& param : graph.parameters().values()) {
    if (!param.requires_grad) {
      continue;
    }

    auto grad_id = graph.gradients().get(param.inst_id);
    if (grad_id.isValid()) {
      graph.insts().emplace(insts::AccumulateGrad(param.inst_id, grad_id));
    }
  }
}

}  // namespace axon
