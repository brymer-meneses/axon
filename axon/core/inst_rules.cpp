module;

#include "llvm/ADT/SmallVector.h"

export module axon.core:inst_rules;

import :inst_kinds;
import :inst;

export namespace axon::core {

class Module;

struct Dependency {
  InstId tensor_id;
  InstId grad_id;
};

// BackwardBuilder is a proxy to the `Module`. It provides clean APIs for
// building the backward graph.
class BackwardWriter {
 public:
  BackwardWriter(Module& module, llvm::SmallVector<Dependency>& deps)
      : module_(module), deps_(deps) {}

  // These functions need to be forward declared to break a cyclic dependency.
  // These are defined in `module.cpp`.
  auto check_requires_grad(InstId inst_id) const -> bool;
  auto get_cached_value(InstId forward_inst_id) -> InstId;
  auto backward(InstId tensor_inst_id, InstId grad_id) -> void;
  auto emit_inst(Inst inst) -> InstId;

 private:
  Module& module_;
  llvm::SmallVector<Dependency>& deps_;
};

template <typename T>
struct InstHandler;

template <>
struct InstHandler<insts::Add> {
  static auto backward(const insts::Add& op, InstId grad_id,
                       BackwardWriter& builder) -> void {
    if (builder.check_requires_grad(op.lhs_id)) {
      builder.backward(op.lhs_id, grad_id);
    }
    if (builder.check_requires_grad(op.rhs_id)) {
      builder.backward(op.rhs_id, grad_id);
    }
  }
};

template <>
struct InstHandler<insts::Mul> {
  static auto backward(const insts::Mul& op, InstId grad_id,
                       BackwardWriter& builder) -> void {
    if (builder.check_requires_grad(op.lhs_id)) {
      auto cached_value_id = builder.get_cached_value(op.rhs_id);
      auto prod = builder.emit_inst(insts::Mul(grad_id, cached_value_id));
      builder.backward(op.lhs_id, prod);
    }

    if (builder.check_requires_grad(op.rhs_id)) {
      auto cached_value_id = builder.get_cached_value(op.lhs_id);
      auto prod = builder.emit_inst(insts::Mul(grad_id, cached_value_id));
      builder.backward(op.rhs_id, prod);
    }
  }
};

template <typename T>
concept HasBackward =
    requires(const T& op, InstId grad_id, BackwardWriter& builder) {
      { InstHandler<T>::backward(op, grad_id, builder) } -> std::same_as<void>;
    };

}  // namespace axon::core
