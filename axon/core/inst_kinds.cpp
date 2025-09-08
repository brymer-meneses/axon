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

struct ExpandDims {
  struct Mapping {
    int64_t dim;
    int64_t scale;
  };

  InstId operand_id;
  llvm::SmallVector<Mapping> mappings;
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

struct Reshape {
  InstId operand_id;
  llvm::SmallVector<int64_t> target_shape;
};

struct Add {
  InstId lhs_id;
  InstId rhs_id;
};

struct Sub {
  InstId lhs_id;
  InstId rhs_id;
};

struct Mul {
  InstId lhs_id;
  InstId rhs_id;
};

struct ScalarMul {
  InstId operand_id;
  double scalar;
};

struct Neg {
  InstId operand_id;
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
      insts::Sub,
      insts::Neg,
      insts::ScalarMul,
      insts::MatMul,
      insts::Sum,
      insts::ExpandDims,
      insts::Unsqueeze,
      insts::Squeeze,
      insts::Transpose,
      insts::AccumulateGrad, 
      insts::Constant,
      insts::GetParameter, 
      insts::OnesLike,
      insts::Reshape
    >;

template <typename InstType>
constexpr bool InstIsUnary =
    llvm::is_one_of<InstType, 
      insts::Sum,
      insts::Squeeze,
      insts::Unsqueeze,
      insts::ExpandDims, 
      insts::Transpose,
      insts::Reshape,
      insts::ScalarMul,
      insts::Neg
  >();

template <typename InstType>
constexpr bool InstIsBinary =
    llvm::is_one_of<InstType, 
      insts::Add,
      insts::MatMul,
      insts::Mul,
      insts::Sub
  >();
// clang-format on:

}  // namespace axon
