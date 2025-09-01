module;

#include "axon/base/dcheck.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:shape_rules;

import :inst_kinds;
import :ids;

export namespace axon {

using Shape = llvm::SmallVector<int64_t>;
using ShapeRef = llvm::ArrayRef<int64_t>;

using ShapeMapping = IdMap<InstId, Shape>;

template <typename T>
struct InferShapeRule;

template <>
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

template <>
struct InferShapeRule<insts::Squeeze> {
  static auto apply(const insts::Squeeze& op, const ShapeMapping& shapes)
      -> Shape {
    AXON_DCHECK(shapes.get(op.operand_id),
                "op.operand_id must have a shape already.");

    Shape shape = shapes.get(op.operand_id)->get();
    AXON_DCHECK(op.dim < static_cast<int32_t>(shape.size()),
                "Dimension must be less than the rank of the operand");
    shape.erase(shape.begin() + op.dim);
    return shape;
  }
};

template <>
struct InferShapeRule<insts::Sum> {
  static auto apply(const insts::Sum& op, const ShapeMapping& shapes) -> Shape {
    AXON_DCHECK(shapes.get(op.operand_id),
                "op.operand_id must have a shape already.");

    Shape shape = shapes.get(op.operand_id)->get();
    auto rank = static_cast<int32_t>(shape.size());

    if (op.keepdims) {
      AXON_DCHECK(op.axis < rank, "Axis must not exceed rank");
      shape[op.axis] = 1;
      return shape;
    }
    shape.erase(shape.begin() + op.axis);
    return shape;
  }
};

template <>
struct InferShapeRule<insts::Broadcast> {
  static auto apply(const insts::Broadcast& op, const ShapeMapping& shapes)
      -> Shape {
    AXON_DCHECK(shapes.get(op.operand_id),
                "op.operand_id must have a shape already.");

    Shape shape = shapes.get(op.operand_id)->get();
    auto rank = static_cast<int32_t>(shape.size());
    for (auto expansion : op.expansions) {
      AXON_DCHECK(expansion.dim < rank, "Dim exceeded the rank");
      shape[expansion.dim] = expansion.scale;
    }
    return shape;
  }
};

template <typename T>
concept HasInferShapeRule = requires(const T& op, const ShapeMapping& shapes) {
  { InferShapeRule<T>::apply(op, shapes) } -> std::same_as<Shape>;
};

}  // namespace axon
