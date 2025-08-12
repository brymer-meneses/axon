
#include "dialect.h"

import std;

namespace axon {

auto AccumulateGradOp::verify() -> mlir::LogicalResult {
  auto requires_grad = getAccumulator().getType().getRequiresGrad();

  return requires_grad
             ? mlir::success()
             : emitOpError("The accumulator needs to require gradients.");
}

auto ConstantOp::print(mlir::OpAsmPrinter& printer) -> void {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"value"});
  printer << getValue();
}

auto ConstantOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
    -> mlir::ParseResult {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes)) {
    return mlir::failure();
  }

  result.addTypes(value.getType());
  return mlir::success();
}

static auto inferReturnType(llvm::ArrayRef<int64_t> lhs_shape,
                            llvm::ArrayRef<int64_t> rhs_shape)
    -> std::optional<llvm::SmallVector<int64_t>> {
  // Handle identical shapes (fast path)
  if (lhs_shape.equals(rhs_shape)) {
    return llvm::SmallVector<int64_t>(lhs_shape.begin(), lhs_shape.end());
  }

  auto lhs_size = static_cast<int32_t>(lhs_shape.size());
  auto rhs_size = static_cast<int32_t>(rhs_shape.size());

  // Get the maximum number of dimensions
  auto max_dims = static_cast<int32_t>(std::max(lhs_size, rhs_size));

  llvm::SmallVector<int64_t> result_shape;
  result_shape.reserve(max_dims);

  // Iterate from the trailing (rightmost) dimensions
  for (auto i = 0; i < max_dims; ++i) {
    // Get dimensions, treating missing dimensions as 1
    auto lhs_dim = (i < lhs_size) ? lhs_shape[lhs_size - 1 - i] : 1;
    auto rhs_dim = (i < rhs_size) ? rhs_shape[rhs_size - 1 - i] : 1;

    // Broadcasting rules:
    // 1. Dimensions of size 1 can be broadcast to any size
    // 2. Dimensions must be equal or one of them must be 1
    if (lhs_dim == rhs_dim) {
      result_shape.push_back(lhs_dim);
    } else if (lhs_dim == 1) {
      result_shape.push_back(rhs_dim);
    } else if (rhs_dim == 1) {
      result_shape.push_back(lhs_dim);
    } else {
      return std::nullopt;
    }
  }

  // Reverse the result since we built it backwards
  std::reverse(result_shape.begin(), result_shape.end());
  return {result_shape};
}

auto AddOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value lhs, mlir::Value rhs) -> void {
  auto lhs_tensor = llvm::cast<mlir::TensorType>(lhs.getType());
  auto rhs_tensor = llvm::cast<mlir::TensorType>(rhs.getType());

  auto inferred_shape =
      inferReturnType(lhs_tensor.getShape(), rhs_tensor.getShape());
  if (!inferred_shape) {
    mlir::emitError(state.location, "incompatible shapes for add operation");
    return;
  }

  auto return_type =
      mlir::RankedTensorType::get(*inferred_shape, lhs_tensor.getElementType());

  build(builder, state, return_type, lhs, rhs);
}

auto AddOp::verify() -> mlir::LogicalResult {
  auto lhs_tensor = getLhs();
  auto rhs_tensor = getLhs();

  return lhs_tensor.getType() == rhs_tensor.getType() ? mlir::success()
                                                      : mlir::failure();
}

auto MulOp::verify() -> mlir::LogicalResult {
  auto lhs_tensor = getLhs();
  auto rhs_tensor = getLhs();

  return lhs_tensor.getType() == rhs_tensor.getType() ? mlir::success()
                                                      : mlir::failure();
}

auto MulOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                  mlir::Value lhs, mlir::Value rhs) -> void {
  auto lhs_tensor = llvm::cast<mlir::TensorType>(lhs.getType());
  auto rhs_tensor = llvm::cast<mlir::TensorType>(rhs.getType());

  auto inferred_shape =
      inferReturnType(lhs_tensor.getShape(), rhs_tensor.getShape());
  if (!inferred_shape) {
    mlir::emitError(state.location, "incompatible shapes for add operation");
    return;
  }

  auto return_type =
      mlir::RankedTensorType::get(*inferred_shape, lhs_tensor.getElementType());
  MulOp::build(builder, state, return_type, lhs, rhs);
}

auto TupleAccessOp::print(mlir::OpAsmPrinter& printer) -> void {
  auto input = getInput();
  auto index = getIndex();

  printer << " ";
  printer << input;
  printer << '[' << index << ']';
  printer << " : " << getResult().getType();
}

auto TupleAccessOp::parse(mlir::OpAsmParser& parser,
                          mlir::OperationState& result) -> mlir::ParseResult {
  mlir::OpAsmParser::UnresolvedOperand input;
  mlir::IntegerAttr indexAttr;
  mlir::Type inputType;
  mlir::Type resultType;

  // Parse the input.
  if (parser.parseOperand(input)) {
    return mlir::failure();
  }

  // Now the [0] part
  if (parser.parseLSquare() ||
      parser.parseAttribute(indexAttr, "index", result.attributes) ||
      parser.parseRSquare()) {
    return mlir::failure();
  }

  if (parser.parseColon() || parser.parseType(resultType)) {
    return mlir::failure();
  }

  if (parser.resolveOperand(input, inputType, result.operands)) {
    return mlir::failure();
  }

  result.addTypes(resultType);
  return mlir::success();
}

}  // namespace axon
