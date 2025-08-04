
#include "dialect.h"

namespace axon {

auto TupleAccessOp::print(mlir::OpAsmPrinter& printer) -> void {
  auto input = getInput();
  auto index = getIndex();

  printer << " ";
  printer << input;
  printer << '[' << index << ']';
  printer << " : " << input.getType().getTypes()[index];
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

static auto print_binary_op(mlir::OpAsmPrinter& printer, mlir::Value lhs,
                            mlir::Value rhs, mlir::Type result) -> void {
  printer << " ";
  printer << lhs << " , " << rhs;
  printer << " : " << result;
}

static auto parse_binary_op(mlir::OpAsmParser& parser,
                            mlir::OperationState& result) -> mlir::ParseResult {
  mlir::OpAsmParser::UnresolvedOperand lhs;
  mlir::OpAsmParser::UnresolvedOperand rhs;
  mlir::Type resultType;

  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs)) {
    return mlir::failure();
  }

  if (parser.parseColon() || parser.parseType(resultType)) {
    return mlir::failure();
  }

  // **This is what you're missing:**
  if (parser.resolveOperand(lhs, resultType, result.operands) ||
      parser.resolveOperand(rhs, resultType, result.operands)) {
    return mlir::failure();
  }

  result.addTypes(resultType);
  return mlir::success();
}

auto AddOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result)
    -> mlir::ParseResult {
  return parse_binary_op(parser, result);
}

auto AddOp::print(mlir::OpAsmPrinter& printer) -> void {
  print_binary_op(printer, getLhs(), getRhs(), getResult().getType());
}

auto MulOp::print(mlir::OpAsmPrinter& printer) -> void {
  print_binary_op(printer, getLhs(), getRhs(), getResult().getType());
}

auto MulOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result)
    -> mlir::ParseResult {
  return parse_binary_op(parser, result);
}
}  // namespace axon
