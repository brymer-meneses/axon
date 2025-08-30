module;

#include "axon/base/dcheck.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:inst_rules;

import :inst_kinds;
import :inst;
import :graph;

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

  auto emit(Inst inst) -> InstId { return graph_.emit(inst); }

  auto accumulateGrad(Dependency dep) -> void {
    if (auto current_grad_id = graph_.gradients().get(dep.inst_id)) {
      dep.grad_id = graph_.emit(insts::Add(current_grad_id, dep.grad_id));
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
      auto prod = ctx.emit(insts::Mul(grad_id, op.rhs_id));
      deps.emplace_back(op.lhs_id, prod);
    }
    if (ctx.checkRequiresGrad(op.rhs_id)) {
      auto prod = ctx.emit(insts::Mul(grad_id, op.lhs_id));
      deps.emplace_back(op.rhs_id, prod);
    }

    return deps;
  }
};

template <>
struct BackwardRule<insts::MatMul> {
  static auto apply(const insts::MatMul& op, InstId grad_id,
                    BackwardContext& ctx) -> llvm::SmallVector<Dependency> {
    llvm::SmallVector<Dependency, 2> deps;
    if (ctx.checkRequiresGrad(op.lhs_id)) {
      auto transposed = ctx.emit(insts::Transpose(op.rhs_id));
      auto prod = ctx.emit(insts::MatMul(grad_id, transposed));
      deps.emplace_back(op.lhs_id, prod);
    }
    if (ctx.checkRequiresGrad(op.rhs_id)) {
      auto transposed = ctx.emit(insts::Transpose(op.lhs_id));
      auto prod = ctx.emit(insts::MatMul(transposed, grad_id));
      deps.emplace_back(op.rhs_id, prod);
    }
    return deps;
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
