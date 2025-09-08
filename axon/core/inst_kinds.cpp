module;

#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:inst_kinds;

import :ids;

export namespace axon {

enum class ShapeInfo {
  Custom,
  None,
  SameAsOperands,
};

struct InstTraits {
  /// number of tensor operands
  int8_t num_operands = 0;
  /// shape inference rule
  ShapeInfo shape_rule = ShapeInfo::Custom;
  /// whether this inst has a backward function
  bool differentiable = false;
};

namespace insts {

struct Sum {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId operand_id;
  // axis to sum
  int32_t axis;

  bool keepdims;
};

struct ExpandDims {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  struct Mapping {
    int64_t dim;
    int64_t scale;
  };

  InstId operand_id;
  llvm::SmallVector<Mapping> mappings;
};

struct Unsqueeze {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId operand_id;
  int32_t dim;
};

struct Squeeze {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId operand_id;
  int32_t dim;
};

struct Transpose {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId operand_id;

  uint32_t from;
  uint32_t to;
};

struct Reshape {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId operand_id;
  llvm::SmallVector<int64_t> target_shape;
};

struct MatMul {
  constexpr static auto traits = InstTraits{
      .num_operands = 2,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId lhs_id;
  InstId rhs_id;
};

struct Add {
  constexpr static auto traits = InstTraits{
      .num_operands = 2,
      .shape_rule = ShapeInfo::SameAsOperands,
      .differentiable = true,
  };

  InstId lhs_id;
  InstId rhs_id;
};

struct Sub {
  constexpr static auto traits = InstTraits{
      .num_operands = 2,
      .shape_rule = ShapeInfo::SameAsOperands,
      .differentiable = true,
  };

  InstId lhs_id;
  InstId rhs_id;
};

struct Mul {
  constexpr static auto traits = InstTraits{
      .num_operands = 2,
      .shape_rule = ShapeInfo::SameAsOperands,
      .differentiable = true,
  };

  InstId lhs_id;
  InstId rhs_id;
};

struct ScalarMul {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::SameAsOperands,
      .differentiable = true,
  };

  InstId operand_id;
  double scalar;
};

struct Neg {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::SameAsOperands,
      .differentiable = true,
  };

  InstId operand_id;
};

struct Constant {
  constexpr static auto traits = InstTraits{
      .num_operands = 0,
      .shape_rule = ShapeInfo::SameAsOperands,
      .differentiable = false,
  };
};

struct AccumulateGrad {
  constexpr static auto traits = InstTraits{
      .num_operands = 2,
      .shape_rule = ShapeInfo::SameAsOperands,
      .differentiable = false,
  };

  InstId inst_id;
  InstId value_id;
};

struct GetParameter {
  constexpr static auto traits = InstTraits{
      .num_operands = 0,
      .shape_rule = ShapeInfo::None,
      .differentiable = false,
  };

  ParamId param_id;
};

struct OnesLike {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::SameAsOperands,
      .differentiable = false,
  };

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

// clang-format on:

}  // namespace axon
