#include <print>
#include <utility>

#include "dialect.h"

namespace axon {

auto ParameterType::print(mlir::AsmPrinter& printer) const -> void {
  printer << "<";
  llvm::interleave(getShape(), printer, "x");
  printer << "x" << getElementType();
  if (getRequiresGrad()) {
    printer << ", requires_grad";
  }
  printer << ">";
}
auto ParameterListType::print(mlir::AsmPrinter& printer) const -> void {
  printer << "<";
  llvm::interleave(getParams(), printer, ", ");
  printer << ">";
}

// TODO: implement this!
auto ParameterType::parse(mlir::AsmParser&) -> mlir::Type { return nullptr; }
auto ParameterListType::parse(mlir::AsmParser&) -> mlir::Type {
  return nullptr;
}

}  // namespace axon
