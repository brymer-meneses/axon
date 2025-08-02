module;

#include "llvm/ADT/SmallVector.h"

export module axon.core:inst_rules;

import :inst_kinds;
import :inst;
import :graph;
import :mod;

export namespace axon {

class BackwardContext {
 public:
  BackwardContext(Module& module) : module_(module) {}

  auto check_requires_grad(InstId inst_id) const -> bool {
    return module_.check_requires_grad(inst_id);
  }

  auto get_cached_value(InstId forward_inst_id) -> InstId {
    if (auto existing_id = cached_values_.get(forward_inst_id);
        existing_id.has_value()) {
      return existing_id;
    }
    auto cached_value_inst =
        module_.forward().get_cached_value(forward_inst_id);
    auto inst_id = module_.backward().emit(cached_value_inst);
    cached_values_.set(forward_inst_id, inst_id);
    return inst_id;
  }

  auto emit(Inst inst) -> InstId { return module_.backward().emit(inst); }

  auto accumulate_grad(InstId forward_inst_id, InstId grad_id) -> void {
    if (auto current_grad_id = module_.gradients().get(forward_inst_id);
        current_grad_id.has_value()) {
      grad_id = emit(insts::Add(current_grad_id, grad_id));
    }

    module_.gradients().set(forward_inst_id, grad_id);
  }

 private:
  Module& module_;
  IdStore<InstId, InstId> cached_values_;
};

struct Dependency {
  InstId inst_id;
  InstId grad_id;
};

template <typename T>
struct BackwardRule;

template <>
struct BackwardRule<insts::Add> {
  static auto apply(const insts::Add& op, InstId grad_id, BackwardContext& ctx)
      -> llvm::SmallVector<Dependency> {
    llvm::SmallVector<Dependency, 2> deps;
    if (ctx.check_requires_grad(op.lhs_id)) {
      deps.emplace_back(op.lhs_id, grad_id);
    }
    if (ctx.check_requires_grad(op.rhs_id)) {
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
    if (ctx.check_requires_grad(op.lhs_id)) {
      auto cached_value_id = ctx.get_cached_value(op.rhs_id);
      auto prod = ctx.emit(insts::Mul(grad_id, cached_value_id));
      deps.emplace_back(op.lhs_id, prod);
    }
    if (ctx.check_requires_grad(op.rhs_id)) {
      auto cached_value_id = ctx.get_cached_value(op.lhs_id);
      auto prod = ctx.emit(insts::Mul(grad_id, cached_value_id));
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
