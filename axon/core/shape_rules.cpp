module;
#include <expected>
#include <print>

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
struct InferShapeRule<insts::Sum> {
  static auto apply(const insts::Sum& op, const ShapeMapping& shapes) -> Shape {
    Shape shape = shapes.get(op.operand_id)->get();
    if (op.keepdims) {
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
    Shape shape = shapes.get(op.operand_id)->get();
    for (auto expansion : op.expansions) {
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
