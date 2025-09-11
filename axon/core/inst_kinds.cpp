module;

#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:inst_kinds;

import :ids;
import :scalar;

export namespace axon {

enum class ShapeInfo {
  Custom,
  None,
  SameAsOperands,
};

struct InstTraits {
  /// number of tensor operands
  i8 num_operands = 0;
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
  i32 axis;

  bool keepdims;
};

struct ExpandDims {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  struct Mapping {
    i64 dim;
    i64 scale;
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
  i32 dim;
};

struct Squeeze {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId operand_id;
  i32 dim;
};

struct Transpose {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId operand_id;

  u32 from;
  u32 to;
};

struct Reshape {
  constexpr static auto traits = InstTraits{
      .num_operands = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId operand_id;
  llvm::SmallVector<i64> target_shape;
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
  Scalar scalar;
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
      .shape_rule = ShapeInfo::None,
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
