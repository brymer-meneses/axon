#include "dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace axon {

auto ListAccessOp::print(mlir::OpAsmPrinter& printer) -> void {
  auto param = getInput();
  auto index = getIndex();

  printer << " ";
  printer << param << ", " << index;
  printer << " : " << getResult().getType();
}

// TODO: implement this!
auto ListAccessOp::parse(mlir::OpAsmParser&, mlir::OperationState&)
    -> mlir::ParseResult {
  return mlir::success();
}

}  // namespace axon
