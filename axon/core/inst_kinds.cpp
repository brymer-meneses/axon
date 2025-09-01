module;

#include <variant>

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

struct Sum {
  InstId operand_id;
  // axis to sum
  int32_t axis;

  bool keepdims;
};

struct Broadcast {
  struct Expansion {
    int32_t dim;
    int32_t scale;
  };

  InstId operand_id;
  llvm::SmallVector<Expansion, 2> expansions;
};

struct Unsqueeze {
  InstId operand_id;
  int32_t dim;
};

struct Squeeze {
  InstId operand_id;
  int32_t dim;
};

struct Transpose {
  InstId operand_id;

  uint32_t from;
  uint32_t to;
};

struct Add {
  InstId lhs_id;
  InstId rhs_id;
};

struct Mul {
  InstId lhs_id;
  InstId rhs_id;
};

struct Constant {};

struct AccumulateGrad {
  InstId inst_id;
  InstId value_id;
};

struct GetParameter {
  ParamId param_id;
};

struct OnesLike {
  InstId operand_id;
};

}  // namespace insts

// clang-format off:
using InstInternalType =
    std::variant<
      insts::Add, 
      insts::Mul, 
      insts::MatMul,
      insts::Sum,
      insts::Broadcast,
      insts::Unsqueeze,
      insts::Squeeze,
      insts::Transpose,
      insts::AccumulateGrad, 
      insts::Constant,
      insts::GetParameter, 
      insts::OnesLike
    >;
// clang-format on:

}  // namespace axon
