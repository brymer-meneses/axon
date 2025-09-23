#include "dialect.h"

#include "generated/dialect.cpp.inc"
#include "generated/dialect_enums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "generated/dialect_type_defs.cpp.inc"

#define GET_OP_CLASSES
#include "generated/dialect_ops.cpp.inc"

namespace axon {

auto AxonDialect::initialize() -> void {
  addTypes<
#define GET_TYPEDEF_LIST
#include "generated/dialect_type_defs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "generated/dialect_ops.cpp.inc"
      >();
}

auto AxonDialect::materializeConstant(mlir::OpBuilder& builder,
                                      mlir::Attribute value, mlir::Type type,
                                      mlir::Location loc)
    -> ::mlir::Operation* {
  auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(value);
  if (!elements) {
    return nullptr;
  }
  return ConstantOp::create(builder, loc, type, elements);
};

}  // namespace axon
