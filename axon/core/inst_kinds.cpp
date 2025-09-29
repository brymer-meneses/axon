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
  SameAsInputs,
};

struct InstTraits {
  /// number of tensor operands
  i8 num_inputs = 0;
  /// shape inference rule
  ShapeInfo shape_rule = ShapeInfo::Custom;
  /// whether this inst has a backward function
  bool differentiable = false;
};

namespace insts {

struct ExpandDims {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  struct Mapping {
    i64 dim;
    i64 scale;

    auto operator==(const Mapping&) const -> bool = default;
  };

  InstId input_id;
  llvm::SmallVector<Mapping> mappings;

  auto operator==(const ExpandDims& other) const -> bool = default;
};

struct Unsqueeze {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId input_id;
  i32 dim;

  auto operator==(const Unsqueeze& other) const -> bool = default;
};

struct Squeeze {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId input_id;
  i32 dim;

  auto operator==(const Squeeze& other) const -> bool = default;
};

struct Transpose {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId input_id;

  u32 from;
  u32 to;

  auto operator==(const Transpose& other) const -> bool = default;
};

struct Reshape {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId input_id;
  llvm::SmallVector<i64> target_shape;

  auto operator==(const Reshape& other) const -> bool = default;
};

struct MatMul {
  constexpr static auto traits = InstTraits{
      .num_inputs = 2,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  InstId lhs_id;
  InstId rhs_id;

  auto operator==(const MatMul& other) const -> bool = default;
};

struct Add {
  constexpr static auto traits = InstTraits{
      .num_inputs = 2,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = true,
  };

  InstId lhs_id;
  InstId rhs_id;

  auto operator==(const Add& other) const -> bool = default;
};

struct Sub {
  constexpr static auto traits = InstTraits{
      .num_inputs = 2,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = true,
  };

  InstId lhs_id;
  InstId rhs_id;

  auto operator==(const Sub& other) const -> bool = default;
};

struct Mul {
  constexpr static auto traits = InstTraits{
      .num_inputs = 2,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = true,
  };

  InstId lhs_id;
  InstId rhs_id;

  auto operator==(const Mul& other) const -> bool = default;
};

struct ScalarMul {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = true,
  };

  InstId input_id;
  Scalar scalar;

  auto operator==(const ScalarMul& other) const -> bool = default;
};

struct Neg {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = true,
  };

  InstId input_id;

  auto operator==(const Neg& other) const -> bool = default;
};

struct Constant {
  constexpr static auto traits = InstTraits{
      .num_inputs = 0,
      .shape_rule = ShapeInfo::None,
      .differentiable = false,
  };

  auto operator==(const Constant&) const -> bool { return true; }
};

struct AccumulateGrad {
  constexpr static auto traits = InstTraits{
      .num_inputs = 2,
      .shape_rule = ShapeInfo::None,
      .differentiable = false,
  };

  InstId sink_id;
  InstId source_id;

  auto operator==(const AccumulateGrad&) const -> bool = default;
};

struct AccumulateData {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::None,
      .differentiable = false,
  };

  auto operator==(const AccumulateData&) const -> bool = default;

  InstId sink_id;
  InstId source_id;
};

struct GetParameter {
  constexpr static auto traits = InstTraits{
      .num_inputs = 0,
      .shape_rule = ShapeInfo::None,
      .differentiable = false,
  };

  ParamId param_id;

  auto operator==(const GetParameter&) const -> bool = default;
};

struct FillLike {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = false,
  };

  InstId input_id;
  Scalar fill_value;

  auto operator==(const FillLike&) const -> bool = default;
};

struct Pow {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = true,
  };

  auto operator==(const Pow&) const -> bool = default;

  InstId input_id;
  Scalar exponent;
};

struct Sum {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  auto operator==(const Sum& other) const -> bool = default;

  InstId input_id;
  // axis to sum
  i32 axis;
  bool keep_dims;
};

struct Mean {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::Custom,
      .differentiable = true,
  };

  auto operator==(const Mean& other) const -> bool = default;

  InstId input_id;
  i64 axis;
  bool keep_dims;
};

struct Softmax {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = true,
  };

  auto operator==(const Softmax&) const -> bool = default;

  InstId input_id;
  i32 axis;
};

struct Relu {
  constexpr static auto traits = InstTraits{
      .num_inputs = 1,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = true,
  };

  auto operator==(const Relu&) const -> bool = default;

  InstId input_id;
};

struct Compare {
  constexpr static auto traits = InstTraits{
      .num_inputs = 2,
      .shape_rule = ShapeInfo::SameAsInputs,
      .differentiable = false,
  };

  enum class Predicate {
    Less,
    LessEq,
    Greater,
    GreaterEq,
    Equal,
    NotEqual,
  };

  auto operator==(const Compare&) const -> bool = default;

  InstId lhs_id;
  InstId rhs_id;
  Predicate predicate;
};

}  // namespace insts

// clang-format off:
using InstInternalType =
    std::variant<
      insts::Add, 
      insts::Mul, 
      insts::Pow,
      insts::Softmax,
      insts::Relu,
      insts::Compare,
      insts::Sub,
      insts::Neg,
      insts::ScalarMul,
      insts::MatMul,
      insts::Sum,
      insts::Mean,
      insts::ExpandDims,
      insts::Unsqueeze,
      insts::Squeeze,
      insts::Transpose,
      insts::AccumulateGrad, 
      insts::AccumulateData, 
      insts::Constant,
      insts::GetParameter, 
      insts::FillLike,
      insts::Reshape
    >;

// clang-format on:

}  // namespace axon
