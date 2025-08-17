#include "dialect.h"

#include "generated/dialect.cpp.inc"

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

}  // namespace axon
