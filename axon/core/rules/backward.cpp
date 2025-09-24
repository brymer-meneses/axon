module;

#include <print>

#include "axon/base/macros.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:backward_rules;

import :graph;
import :inst_kinds;
import :inst;

namespace axon {

export struct Dependency {
  InstId inst_id;
  InstId grad_id;
};

export class BackwardContext {
 public:
  BackwardContext(Graph& graph) : graph_(graph) {}

  auto checkRequiresGrad(InstId inst_id) const -> bool {
    return graph_.checkRequiresGrad(inst_id);
  }

  auto createOp(Inst&& inst) -> InstId {
    return graph_.createOp(std::move(inst), /*emit_grad=*/false);
  }

  auto setCurrentInstId(InstId inst_id) -> void { current_inst_id_ = inst_id; }
  auto getCurrentInstId() const -> InstId { return current_inst_id_; }

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
  InstId current_inst_id_ = InstId::None;
};

// Expand a reduced tensor along `axis` to match the size of `ref_id` at that
// axis. Assumes the input has keepdims=1 at `axis`.
static auto expandAlongAxisToMatch(BackwardContext& ctx, InstId inst_id,
                                   u64 axis, InstId ref_id) -> InstId {
  llvm::SmallVector<insts::ExpandDims::Mapping, 1> mappings;
  auto inst_shape = ctx.getShape(inst_id);
  auto ref_shape = ctx.getShape(ref_id);

  AXON_DCHECK(axis < ref_shape.size());
  AXON_DCHECK(inst_shape[axis] == 1);

  mappings.push_back({.dim = static_cast<i64>(axis), .scale = ref_shape[axis]});
  return ctx.createOp(insts::ExpandDims(inst_id, std::move(mappings)));
}

export template <typename T>
struct BackwardRule;

export template <>
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

export template <>
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

export template <>
struct BackwardRule<insts::Sub> {
  static auto apply(const insts::Sub& op, InstId grad_id, BackwardContext& ctx)
      -> llvm::SmallVector<Dependency> {
    llvm::SmallVector<Dependency, 2> deps;
    if (ctx.checkRequiresGrad(op.lhs_id)) {
      deps.emplace_back(op.lhs_id, grad_id);
    }

    if (ctx.checkRequiresGrad(op.rhs_id)) {
      auto negated = ctx.createOp(insts::Neg(grad_id));
      deps.emplace_back(op.rhs_id, negated);
    }

    return deps;
  }
};

export template <>
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
struct BackwardRule<insts::ExpandDims> {
  static auto apply(const insts::ExpandDims& op, InstId grad_id,
                    BackwardContext& ctx) -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }
    for (auto mapping : op.mappings) {
      grad_id =
          ctx.createOp(insts::Sum(grad_id, mapping.dim, /*keepdims=*/true));
    }
    return {{op.operand_id, grad_id}};
  }
};

export template <>
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

export template <>
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

export template <>
struct BackwardRule<insts::Neg> {
  static auto apply(const insts::Neg& op, InstId grad_id, BackwardContext& ctx)
      -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }

    grad_id = ctx.createOp(insts::Neg(grad_id));
    return {{op.operand_id, grad_id}};
  }
};

export template <>
struct BackwardRule<insts::ScalarMul> {
  static auto apply(const insts::ScalarMul& op, InstId grad_id,
                    BackwardContext& ctx) -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }

    grad_id = ctx.createOp(insts::ScalarMul(grad_id, op.scalar));
    return {{op.operand_id, grad_id}};
  }
};

export template <>
struct BackwardRule<insts::Sum> {
  static auto apply(const insts::Sum& op, InstId grad_id, BackwardContext& ctx)
      -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }

    if (!op.keep_dims) {
      grad_id = ctx.createOp(insts::Unsqueeze(grad_id, op.axis));
    }

    grad_id = expandAlongAxisToMatch(ctx, grad_id, op.axis, op.operand_id);
    return {{op.operand_id, grad_id}};
  }
};

export template <>
struct BackwardRule<insts::Pow> {
  static auto apply(const insts::Pow& op, InstId grad_id, BackwardContext& ctx)
      -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }

    auto pow_grad = ctx.createOp(insts::Pow(grad_id, op.exponent - 1.0));
    auto scalar_grad = ctx.createOp(insts::ScalarMul(pow_grad, op.exponent));

    return {{op.operand_id, scalar_grad}};
  }
};

export template <>
struct BackwardRule<insts::Softmax> {
  static auto apply(const insts::Softmax& op, InstId grad_id,
                    BackwardContext& ctx) -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }

    // Use the current op's output as the softmax activation `y` to avoid
    // recomputation: dx = y * (g - sum(g * y, axis, keepdims=True))
    auto y_id = ctx.getCurrentInstId();

    auto gy = ctx.createOp(insts::Mul(grad_id, y_id));
    auto sum_gy = ctx.createOp(insts::Sum(gy, op.axis, /*keepdims=*/true));

    // Expand the reduced axis back to match the gradient's shape so that
    // binary ops have identical shapes.
    auto sum_gy_expanded = expandAlongAxisToMatch(ctx, sum_gy, op.axis, y_id);

    auto sub = ctx.createOp(insts::Sub(grad_id, sum_gy_expanded));
    auto dx = ctx.createOp(insts::Mul(y_id, sub));

    return {{op.operand_id, dx}};
  }
};

export template <>
struct BackwardRule<insts::Relu> {
  static auto apply(const insts::Relu& op, InstId grad_id, BackwardContext& ctx)
      -> llvm::SmallVector<Dependency> {
    if (!ctx.checkRequiresGrad(op.operand_id)) {
      return {};
    }

    auto predicate = insts::Compare::Predicate::Greater;
    auto zeros_like =
        ctx.createOp(insts::FillLike(op.operand_id, Scalar(0.0f)));
    auto mask =
        ctx.createOp(insts::Compare(op.operand_id, zeros_like, predicate));
    auto masked = ctx.createOp(insts::Mul(mask, grad_id));

    return {{op.operand_id, masked}};
  }
};

export template <typename T>
concept HasBackwardRule =
    requires(const T& op, InstId grad_id, BackwardContext& ctx) {
      {
        BackwardRule<T>::apply(op, grad_id, ctx)
      } -> std::same_as<llvm::SmallVector<Dependency>>;
    };

}  // namespace axon
