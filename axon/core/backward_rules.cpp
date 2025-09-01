module;

#include <print>

#include "llvm/ADT/SmallVector.h"

export module axon.core:backward_rules;

import :graph;
import :inst_kinds;
import :inst;

export namespace axon {

struct Dependency {
  InstId inst_id;
  InstId grad_id;
};

class BackwardContext {
 public:
  BackwardContext(Graph& graph) : graph_(graph) {}

  auto checkRequiresGrad(InstId inst_id) const -> bool {
    return graph_.checkRequiresGrad(inst_id);
  }

  auto createOp(Inst inst) -> InstId {
    return graph_.createOp(inst, /*emit_grad=*/false);
  }

  auto accumulateGrad(Dependency dep) -> void {
    if (auto current_grad_id = graph_.gradients().get(dep.inst_id)) {
      dep.grad_id = createOp(insts::Add(current_grad_id, dep.grad_id));
    }
    graph_.gradients().set(dep.inst_id, dep.grad_id);
  }

 private:
  Graph& graph_;
};

template <typename T>
struct BackwardRule;

template <>
struct BackwardRule<insts::Add> {
  static auto apply(const insts::Add& op, InstId grad_id, BackwardContext& ctx)
      -> llvm::SmallVector<Dependency> {
    llvm::SmallVector<Dependency, 2> deps;
    if (ctx.checkRequiresGrad(op.lhs_id)) {
      deps.emplace_back(op.lhs_id, grad_id);
    }
    if (ctx.checkRequiresGrad(op.rhs_id)) {
      deps.emplace_back(op.rhs_id, grad_id);
    }

    return deps;
  }
};

template <>
struct BackwardRule<insts::Mul> {
  static auto apply(const insts::Mul& op, InstId grad_id, BackwardContext& ctx)
      -> llvm::SmallVector<Dependency> {
    llvm::SmallVector<Dependency, 2> deps;
    if (ctx.checkRequiresGrad(op.lhs_id)) {
      auto prod = ctx.createOp(insts::Mul(grad_id, op.rhs_id));
      deps.emplace_back(op.lhs_id, prod);
    }
    if (ctx.checkRequiresGrad(op.rhs_id)) {
      auto prod = ctx.createOp(insts::Mul(grad_id, op.lhs_id));
      deps.emplace_back(op.rhs_id, prod);
    }

    return deps;
  }
};

template <>
struct BackwardRule<insts::Broadcast> {
  static auto apply(const insts::Broadcast& op, InstId grad_id,
                    BackwardContext& ctx) -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }
    for (auto expansion : op.expansions) {
      grad_id =
          ctx.createOp(insts::Sum(grad_id, expansion.dim, /*keepdims=*/true));
    }
    return {{op.operand_id, grad_id}};
  }
};

template <>
struct BackwardRule<insts::Unsqueeze> {
  static auto apply(const insts::Unsqueeze& op, InstId grad_id,
                    BackwardContext& ctx) -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }

    grad_id = ctx.createOp(insts::Squeeze(grad_id, op.dim));
    return {{op.operand_id, grad_id}};
  }
};

template <typename T>
concept HasBackwardRule =
    requires(const T& op, InstId grad_id, BackwardContext& ctx) {
      {
        BackwardRule<T>::apply(op, grad_id, ctx)
      } -> std::same_as<llvm::SmallVector<Dependency>>;
    };

}  // namespace axon
