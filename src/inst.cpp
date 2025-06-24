module;

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

export module axon.inst;

import axon.ids;
import axon.storage;

namespace axon {

namespace insts {

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

}  // namespace insts

export struct Dependency {
  using GradOp = std::variant<insts::AddBackward, insts::MulBackward,
                              insts::MatMulBackwardL, insts::MatMulBackwardR>;

  GradOp op;

  // The insruction that requires this gradient.
  InstId inst_id;
};

export struct Inst {
  using Op = std::variant<insts::Create, insts::Add, insts::MatMul, insts::Mul>;

  Op op;
  DataId data_id;
  DataId grad_id = DataId::Invalid;
  bool is_observed = false;
  llvm::SmallVector<Dependency, 2> deps{};

  auto requires_grad() const -> bool {
    return grad_id.has_value() or grad_id == DataId::Pending;
  }
};

}  // namespace axon
