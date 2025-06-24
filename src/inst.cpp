module;

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

export module axon.inst;

import axon.ids;
import axon.storage;

namespace axon {

export struct AddBackward {};

export struct MulBackward {
  DataId data;
};

export struct MatMulBackwardL {
  DataId right_id;
};

export struct MatMulBackwardR {
  DataId left_id;
};

export struct Dependency {
  using GradOp =
      std::variant<AddBackward, MulBackward, MatMulBackwardL, MatMulBackwardR>;

  GradOp op;

  // The insruction that requires this gradient.
  InstId inst_id;
};

export struct Create {};

export struct Add {
  InstId lhs_id;
  InstId rhs_id;
};

export struct Mul {
  InstId lhs_id;
  InstId rhs_id;
};

export struct MatMul {
  InstId lhs_id;
  InstId rhs_id;
};

export struct Inst {
  using Op = std::variant<Create, Add, MatMul, Mul>;

  Op op;
  DataId data_id;
  DataId grad_id = DataId::Invalid;
  bool is_observed = false;
  llvm::SmallVector<Dependency, 2> deps{};

  auto requires_grad() const -> bool {
    return grad_id.has_value() or grad_id == DataId::Pending;
  }
};

export struct BackwardBuilder {
  using InstStorage = Storage<InstId, Inst>;

  static auto build(const Add& op, Inst& inst, const InstStorage& storage)
      -> void {
    const auto& lhs = storage.get(op.lhs_id);
    const auto& rhs = storage.get(op.rhs_id);

    if (lhs.requires_grad() or rhs.requires_grad()) {
      inst.grad_id = DataId::Pending;
    }

    if (lhs.requires_grad()) {
      inst.deps.emplace_back(AddBackward(), op.lhs_id);
    }
    if (rhs.requires_grad()) {
      inst.deps.emplace_back(AddBackward(), op.rhs_id);
    }
  };

  static auto build(const Mul& op, Inst& inst, const InstStorage& storage)
      -> void {
    const auto& lhs = storage.get(op.lhs_id);
    const auto& rhs = storage.get(op.rhs_id);

    if (lhs.requires_grad() or rhs.requires_grad()) {
      inst.grad_id = DataId::Pending;
    }

    if (lhs.requires_grad()) {
      inst.deps.emplace_back(MulBackward(rhs.data_id), op.lhs_id);
    }

    if (rhs.requires_grad()) {
      inst.deps.emplace_back(MulBackward(lhs.data_id), op.rhs_id);
    }
  };

  static auto build(const MatMul& op, Inst& inst, const InstStorage& storage)
      -> void {
    const auto& lhs = storage.get(op.lhs_id);
    const auto& rhs = storage.get(op.rhs_id);

    if (lhs.requires_grad() or rhs.requires_grad()) {
      inst.grad_id = DataId::Pending;
    }

    if (lhs.requires_grad()) {
      inst.deps.emplace_back(MatMulBackwardL(rhs.data_id), op.lhs_id);
    }

    if (rhs.requires_grad()) {
      inst.deps.emplace_back(MatMulBackwardR(lhs.data_id), op.rhs_id);
    }
  };

  static auto build(const Create& create, Inst& inst, InstStorage& storage)
      -> void {};
};

}  // namespace axon
