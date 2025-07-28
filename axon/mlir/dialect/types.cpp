#include "dialect.h"

namespace axon {

auto TensorRefType::print(mlir::AsmPrinter& printer) const -> void {
  printer << "<";
  if (isDynamic()) {
    printer << "*";
  } else {
    llvm::interleave(getShape(), printer, "x");
  }

  printer << "x" << getElementType();
  if (getRequiresGrad()) {
    printer << ", rg";
  }
  printer << ">";
}
auto TensorRefListType::print(mlir::AsmPrinter& printer) const -> void {
  printer << "<";
  llvm::interleave(getValues(), printer, ", ");
  printer << ">";
}

// TODO: implement this!
auto TensorRefType::parse(mlir::AsmParser&) -> mlir::Type { return nullptr; }
auto TensorRefListType::parse(mlir::AsmParser&) -> mlir::Type {
  return nullptr;
}

}  // namespace axon
