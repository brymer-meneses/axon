#include "dialect.h"

namespace axon {

auto TensorRefType::print(mlir::AsmPrinter& printer) const -> void {
  printer << "<";
  llvm::interleave(getShape(), printer, "x");
  printer << "x" << getElementType();
  if (getRequiresGrad()) {
    printer << ", requires_grad";
  }
  printer << ">";
}

auto TensorRefType::parse(mlir::AsmParser& parser) -> mlir::Type {
  if (parser.parseLess()) {
    return {};
  }

  llvm::SmallVector<int64_t> shape;
  mlir::Type element_type;
  bool requires_grad = false;

  if (parser.parseDimensionList(shape)) {
    return {};
  }

  if (parser.parseType(element_type)) {
    return {};
  }

  // Check for optional ", requires_grad"
  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseKeyword("requires_grad")) {
      return {};
    }
    requires_grad = true;
  }

  // Parse closing '>'
  if (parser.parseGreater()) {
    return {};
  }

  // Create and return the TensorRefType
  return TensorRefType::get(parser.getContext(), element_type, shape,
                            requires_grad);
}

}  // namespace axon
