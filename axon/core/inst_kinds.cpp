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

struct Constant {
  DataId data_id;
};

struct AccumulateGrad {
  InstId inst_id;
  InstId value_id;
};

struct GetParameter {
  ParamId param_id;
};

}  // namespace insts

using InstInternalType =
    std::variant<insts::Add, insts::Mul, insts::AccumulateGrad, insts::Constant,
                 insts::GetParameter>;

template <typename T>
constexpr bool IsExpressionInst =
    llvm::is_one_of<T, insts::Add, insts::Mul, insts::MatMul>();

}  // namespace axon
