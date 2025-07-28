module;

#include "llvm/ADT/STLExtras.h"

export module axon.core:inst_kinds;

import :ids;

export namespace axon {

namespace insts {

struct MatMul {
  InstId lhs_id;
  InstId rhs_id;
};

struct Transpose {
  InstId value_id;
};

struct Add {
  InstId lhs_id;
  InstId rhs_id;
};

struct Mul {
  InstId lhs_id;
  InstId rhs_id;
};

struct GetInput {
  InputId input_id;
};

struct GetCachedValue {
  CachedValueId cached_value_id;
};

struct SetCachedValue {
  CachedValueId cached_value_id;

  // The value to set to the cached_value_id.
  InstId new_value_id;
};

struct AccumulateGrad {
  InputId input_id;
  InstId value_id;
};

struct InitialGradient {};

struct LocalTensor {};

struct ModuleCall {};

}  // namespace insts

using InstInternalType =
    std::variant<insts::Add, insts::Mul, insts::GetInput, insts::GetCachedValue,
                 insts::SetCachedValue, insts::LocalTensor,
                 insts::InitialGradient, insts::AccumulateGrad>;

template <typename T>
constexpr bool IsExpressionInst =
    llvm::is_one_of<T, insts::Add, insts::Mul, insts::MatMul>();

}  // namespace axon
