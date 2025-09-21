module;

#include <utility>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:shape_rules;

import :inst_kinds;
import :ids;

namespace axon {

export using Shape = llvm::SmallVector<i64>;
export using ShapeRef = llvm::ArrayRef<i64>;

export using ShapeMapping = IdMap<InstId, Shape>;

export template <typename T>
struct InferShapeRule;

export template <>
struct InferShapeRule<insts::MatMul> {
  static auto apply(const insts::MatMul& op, const ShapeMapping& shapes)
      -> Shape {
    AXON_DCHECK(shapes.get(op.lhs_id), "op.lhs_id must have a shape already.");
    AXON_DCHECK(shapes.get(op.rhs_id), "op.rhs_id must have a shape already.");

    auto lhs_shape = shapes.get(op.lhs_id)->get();
    auto rhs_shape = shapes.get(op.rhs_id)->get();

    AXON_DCHECK(lhs_shape.size() == rhs_shape.size(),
                "lhs and rhs must have the same rank.");

    Shape shape = lhs_shape;
    // (N, A, B) @ (N, B, C) => (N, A, C)
    if (lhs_shape.size() == 3) {
      shape[1] = lhs_shape[1];
      shape[2] = rhs_shape[2];
      return shape;
    }

    // (A, B) @ (B, C) => (N, A, C)
    if (lhs_shape.size() == 2) {
      shape[0] = lhs_shape[0];
      shape[1] = rhs_shape[1];
      return shape;
    }

    AXON_UNREACHABLE("MatMul should only have ranks of 2 or 3");
  }
};

export template <>
struct InferShapeRule<insts::Transpose> {
  static auto apply(const insts::Transpose& op, const ShapeMapping& shapes)
      -> Shape {
    AXON_DCHECK(shapes.get(op.operand_id),
                "op.operand_id must have a shape already.");

    Shape shape = shapes.get(op.operand_id)->get();
    AXON_DCHECK(op.from < shape.size(),
                "op.from must not exceed the rank of the tensor");
    AXON_DCHECK(op.to < shape.size(),
                "op.to must not exceed the rank of the tensor");
    std::swap(shape[op.from], shape[op.to]);
    return shape;
  }
};

export template <>
struct InferShapeRule<insts::Unsqueeze> {
  static auto apply(const insts::Unsqueeze& op, const ShapeMapping& shapes)
      -> Shape {
    AXON_DCHECK(shapes.get(op.operand_id),
                "op.operand_id must have a shape already.");

    Shape shape = shapes.get(op.operand_id)->get();
    shape.insert(shape.begin() + op.dim, 1);
    return shape;
  }
};

export template <>
struct InferShapeRule<insts::Squeeze> {
  static auto apply(const insts::Squeeze& op, const ShapeMapping& shapes)
      -> Shape {
    AXON_DCHECK(shapes.get(op.operand_id),
                "op.operand_id must have a shape already.");

    Shape shape = shapes.get(op.operand_id)->get();
    AXON_DCHECK(op.dim < static_cast<i32>(shape.size()),
                "Dimension must be less than the rank of the operand");
    shape.erase(shape.begin() + op.dim);
    return shape;
  }
};

template <typename T>
static auto inferShapeOfReduceInst(const T& op, const ShapeMapping& shapes)
    -> Shape {
  AXON_DCHECK(shapes.get(op.operand_id),
              "op.operand_id must have a shape already.");

  Shape shape = shapes.get(op.operand_id)->get();
  auto rank = static_cast<i32>(shape.size());

  if (op.keep_dims) {
    AXON_DCHECK(op.axis < rank, "Axis must not exceed rank");
    shape[op.axis] = 1;
    return shape;
  }
  shape.erase(shape.begin() + op.axis);
  return shape;
}

export template <>
struct InferShapeRule<insts::Sum> {
  static auto apply(const insts::Sum& op, const ShapeMapping& shapes) -> Shape {
    return inferShapeOfReduceInst(op, shapes);
  }
};

export template <>
struct InferShapeRule<insts::ExpandDims> {
  static auto apply(const insts::ExpandDims& op, const ShapeMapping& shapes)
      -> Shape {
    AXON_DCHECK(shapes.get(op.operand_id),
                "op.operand_id must have a shape already.");

    Shape shape = shapes.get(op.operand_id)->get();
    auto rank = static_cast<i32>(shape.size());
    for (auto expansion : op.mappings) {
      AXON_DCHECK(expansion.dim < rank, "Dim {} exceeded the rank {}",
                  expansion.dim, rank);
      shape[expansion.dim] = expansion.scale;
    }
    return shape;
  }
};

export template <>
struct InferShapeRule<insts::Reshape> {
  static auto apply(const insts::Reshape& op, const ShapeMapping& shapes)
      -> Shape {
    AXON_DCHECK(shapes.get(op.operand_id),
                "op.operand_id must have a shape already.");
    return {op.target_shape.begin(), op.target_shape.end()};
  }
};

export template <typename T>
concept HasInferShapeRule = requires(const T& op, const ShapeMapping& shapes) {
  { InferShapeRule<T>::apply(op, shapes) } -> std::same_as<Shape>;
};

}  // namespace axon
