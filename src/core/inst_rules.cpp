module;

#include "llvm/ADT/SmallVector.h"

export module axon.core:inst_rules;

import :inst_kinds;
import :inst;

export namespace axon {

class Module;

struct Dependency {
  InstId tensor_id;
  InstId grad_id;
};

class BackwardBuilder {
 public:
  BackwardBuilder(Module& module, llvm::SmallVector<Dependency>& deps)
      : module_(module), deps_(deps) {}

  auto check_requires_grad(InstId inst_id) const -> bool;
  auto get_cached_value(InstId forward_inst_id) -> InstId;
  auto track(InstId tensor_inst_id, InstId grad_id) -> void;
  auto emit_inst(Inst inst) -> InstId;

 private:
  Module& module_;
  llvm::SmallVector<Dependency>& deps_;
};

template <typename T>
struct InstHandler {
  auto backward(const T&, InstId, BackwardBuilder&) -> void {
    static_assert(false, "Base case");
  }
};

template <>
struct InstHandler<insts::Add> {
  auto backward(const insts::Add& op, InstId grad_id, BackwardBuilder& builder)
      -> void {
    if (builder.check_requires_grad(op.lhs_id)) {
      builder.track(op.lhs_id, grad_id);
    }
    if (builder.check_requires_grad(op.rhs_id)) {
      builder.track(op.rhs_id, grad_id);
    }
  }
};

template <>
struct InstHandler<insts::Mul> {
  auto backward(const insts::Mul& op, InstId grad_id, BackwardBuilder& builder)
      -> void {
    if (builder.check_requires_grad(op.lhs_id)) {
      auto cached_value_id = builder.get_cached_value(op.rhs_id);
      auto prod = builder.emit_inst(insts::Mul(grad_id, cached_value_id));
      builder.track(op.lhs_id, prod);
    }

    if (builder.check_requires_grad(op.rhs_id)) {
      auto cached_value_id = builder.get_cached_value(op.lhs_id);
      auto prod = builder.emit_inst(insts::Mul(grad_id, cached_value_id));
      builder.track(op.rhs_id, prod);
    }
  }
};

template <typename T>
concept HasBackward =
    requires(const T& op, InstId grad_id, BackwardBuilder& builder) {
      { InstHandler<T>::backward(op, grad_id, builder) } -> std::same_as<void>;
    };

}  // namespace axon
