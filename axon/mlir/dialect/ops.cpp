#include "dialect.h"

namespace axon {

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

auto ConstantOp::print(mlir::OpAsmPrinter& printer) -> void {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

}  // namespace axon
