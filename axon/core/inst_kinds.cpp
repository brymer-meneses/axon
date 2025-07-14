module;

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

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
  CachedValueId value_id;
};

struct SetCachedValue {
  InstId value;
  CachedValueId value_id;
};

struct Constant {};

}  // namespace insts

using InstInternalType =
    std::variant<insts::MatMul, insts::Add, insts::Mul, insts::Transpose,
                 insts::GetInput, insts::GetCachedValue, insts::SetCachedValue,
                 insts::Constant>;

template <typename T>
constexpr bool IsExpressionInst =
    llvm::is_one_of<T, insts::Add, insts::Mul, insts::MatMul>();

}  // namespace axon
