
#include <optional>

#include "dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

namespace axon {

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

  static constexpr auto compute_num_elems = [](llvm::ArrayRef<int64_t> shape) {
    int64_t elems = 1;
    for (auto dim : shape) {
      elems *= dim;
    }
    return elems;
  };

  auto lhs_elems = compute_num_elems(target_shape);
  auto rhs_elems = compute_num_elems(operand.getShape());

  if (lhs_elems != rhs_elems) {
    emitOpError() << "Not a valid reshape.";
    return mlir::failure();
  }

  return mlir::success();
}

auto MatMulOp::verify() -> mlir::LogicalResult {
  auto lhs = mlir::cast<mlir::RankedTensorType>(getLhs().getType());
  auto rhs = mlir::cast<mlir::RankedTensorType>(getRhs().getType());

  if (lhs.getRank() != rhs.getRank()) {
    emitOpError() << "Operands must match ranks";
    return mlir::failure();
  }

  auto lhs_shape = lhs.getShape();
  auto rhs_shape = rhs.getShape();

  if (lhs.getRank() == 3) {
    if (lhs_shape[2] != rhs_shape[1]) {
      emitOpError() << std::format(
          "Cannot perform matrix multiplication on tensors of {} and {}.",
          lhs_shape, rhs_shape);
    }

    return mlir::success();
  }

  if (lhs.getRank() == 2) {
    if (lhs_shape[1] != rhs_shape[0]) {
      emitOpError() << std::format(
          "Cannot perform matrix multiplication on tensors of {} and {}.",
          lhs_shape, rhs_shape);
    }
    return mlir::success();
  }

  return mlir::failure();
}

auto AccumulateGradOp::verify() -> mlir::LogicalResult {
  auto requires_grad = getAccumulator().getType().getRequiresGrad();

  auto accum_shape = getAccumulator().getType().getShape();
  auto value_shape = getValue().getType().getShape();

  if (!requires_grad) {
    emitOpError() << "The accumulator must require gradients";
    return mlir::failure();
  }

  if (accum_shape != value_shape) {
    emitOpError()
        << "The shape of the value tensor must be equal to the accumulator";
    return mlir::failure();
  }

  return mlir::success();
}

void AccumulateGradOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance>& effects) {
  // Operand 0: accumulator (written to)
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getAccumulatorMutable());

  // Operand 1: value (read)
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getValueMutable());
}

}  // namespace axon
