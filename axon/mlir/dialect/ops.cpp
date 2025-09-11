
#include <optional>
#include <utility>

#include "dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

import axon.mlir;
import axon.base;

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
  static constexpr auto compute_num_elems = [](llvm::ArrayRef<int64_t> shape) {
    int64_t elems = 1;
    for (auto dim : shape) {
      elems *= dim;
    }
    return elems;
  };

  auto operand = mlir::cast<mlir::RankedTensorType>(getOperand().getType());
  auto source_elems = compute_num_elems(operand.getShape());
  auto target_elems = compute_num_elems(getTargetShape());

  if (source_elems != target_elems) {
    emitOpError() << std::format("{} -> {} is not a valid reshape",
                                 operand.getShape(), getTargetShape());
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

  static auto transpose = [&](llvm::SmallVector<i64>& vec) {
    if (vec.size() == 3) {
      std::swap(vec[1], vec[2]);
    } else {
      std::swap(vec[0], vec[1]);
    }
  };

  llvm::SmallVector<i64> lhs_shape(lhs.getShape());
  llvm::SmallVector<i64> rhs_shape(rhs.getShape());
  if (getTransposeLhs()) {
    transpose(lhs_shape);
  }
  if (getTransposeRhs()) {
    transpose(rhs_shape);
  }

  if (lhs.getRank() == 3) {
    if (lhs_shape[2] != rhs_shape[1]) {
      emitOpError() << std::format(
          "Cannot perform matrix multiplication on tensors of {} and {}.",
          lhs_shape, rhs_shape);
      return mlir::failure();
    }

    return mlir::success();
  }

  if (lhs.getRank() == 2) {
    if (lhs_shape[1] != rhs_shape[0]) {
      emitOpError() << std::format(
          "Cannot perform matrix multiplication on tensors of {} and {}.",
          lhs_shape, rhs_shape);
      return mlir::failure();
    }
    return mlir::success();
  }

  return mlir::failure();
}

auto AccumulateOp::verify() -> mlir::LogicalResult {
  auto sink_shape = getSink().getType().getShape();
  auto source_shape = getSource().getType().getShape();

  if (sink_shape != source_shape) {
    emitOpError()
        << "The shape of the value tensor must be equal to the accumulator";
    return mlir::failure();
  }

  return mlir::success();
}

}  // namespace axon
