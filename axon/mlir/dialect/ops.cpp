
#include <optional>

#include "dialect.h"

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

auto ReshapeOp::verify() -> mlir::LogicalResult {
  auto target_shape = getTargetShape();
  auto operand = mlir::cast<mlir::RankedTensorType>(getOperand().getType());

  if (static_cast<int64_t>(target_shape.size()) != operand.getRank()) {
    return mlir::failure();
  }

  static constexpr auto compute_num_elems = [](llvm::ArrayRef<int64_t> shape) {
    int64_t elems = 1;
    for (auto dim : shape) {
      elems *= -dim;
    }
    return elems;
  };

  auto lhs_elems = compute_num_elems(target_shape);
  auto rhs_elems = compute_num_elems(operand.getShape());
  return mlir::success(lhs_elems == rhs_elems);
}

}  // namespace axon
