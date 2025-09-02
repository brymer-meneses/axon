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

  auto getShape(InstId inst_id) -> llvm::ArrayRef<int64_t> {
    return graph_.getShape(inst_id);
  }

 private:
  Graph& graph_;
};

template <typename T>
struct BackwardRule;

template <>
struct BackwardRule<insts::MatMul> {
  static auto apply(const insts::MatMul& op, InstId grad_id,
                    BackwardContext& ctx) -> llvm::SmallVector<Dependency> {
    llvm::SmallVector<Dependency, 2> deps;

    // Suppose C = A @ B
    //
    // A: (M, N)
    // B: (N, L)
    // C: (M, L)
    // G: (M, L)
    //
    // We want to combine G (which is the local gradient) with the A and B. dA
    // and dB are the gradients of some function L with respect to
    // A and B respectively.
    //
    // dA: G @ B^T  (M, L) @ (N, L)^T => (M, N)
    // dB: A^T @ G  (M, N)^T @ (M, L) => (N, L)

    auto transpose = [&](InstId inst_id) -> InstId {
      auto shape = ctx.getShape(inst_id);
      if (shape.size() == 3) {
        return ctx.createOp(insts::Transpose(inst_id, 1, 2));
      }
      return ctx.createOp(insts::Transpose(inst_id, 0, 1));
    };

    if (ctx.checkRequiresGrad(op.lhs_id)) {
      auto rhs_t = transpose(op.rhs_id);
      auto new_grad = ctx.createOp(insts::MatMul(grad_id, rhs_t));
      deps.emplace_back(op.lhs_id, new_grad);
    }

    if (ctx.checkRequiresGrad(op.rhs_id)) {
      auto lhs_t = transpose(op.lhs_id);
      auto new_grad = ctx.createOp(insts::MatMul(lhs_t, grad_id));
      deps.emplace_back(op.rhs_id, new_grad);
    }

    return deps;
  }
};

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

template <>
struct BackwardRule<insts::Reshape> {
  static auto apply(const insts::Reshape& op, InstId grad_id,
                    BackwardContext& ctx) -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }

    llvm::SmallVector<int64_t> target_shape(ctx.getShape(op.operand_id));
    grad_id = ctx.createOp(insts::Reshape(grad_id, std::move(target_shape)));
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
