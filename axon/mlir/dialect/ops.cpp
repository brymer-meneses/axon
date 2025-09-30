
#include <optional>
#include <utility>

#include "axon/base/macros.h"
#include "dialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

auto ReshapeOp::verify() -> mlir::LogicalResult {
  static constexpr auto compute_num_elems = [](llvm::ArrayRef<int64_t> shape) {
    int64_t elems = 1;
    for (auto dim : shape) {
      elems *= dim;
    }
    return elems;
  };

  auto input = mlir::cast<mlir::RankedTensorType>(getInput().getType());
  auto source_elems = compute_num_elems(input.getShape());
  auto target_elems = compute_num_elems(getTargetShape());

  if (source_elems != target_elems) {
    emitOpError() << std::format("{} -> {} is not a valid reshape",
                                 input.getShape(), getTargetShape());
    return mlir::failure();
  }

  return mlir::success();
}

auto MatMulOp::verify() -> mlir::LogicalResult {
  auto lhs = mlir::cast<mlir::RankedTensorType>(getLhs().getType());
  auto rhs = mlir::cast<mlir::RankedTensorType>(getRhs().getType());

  if (lhs.getRank() != rhs.getRank()) {
    emitOpError() << "inputs must match ranks";
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
    emitOpError() << std::format(
        "the shape of the value tensor must match the accumulator "
        "expected {} got {}.",
        sink_shape, source_shape);
    return mlir::failure();
  }

  return mlir::success();
}

}  // namespace axon
