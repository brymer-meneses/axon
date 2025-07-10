#include "dialect.h"

#include "generated/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "generated/DialectTypeDefs.cpp.inc"

#define GET_OP_CLASSES
#include "generated/DialectOps.cpp.inc"
namespace axon {

auto AxonDialect::initialize() -> void {
  addTypes<
#define GET_TYPEDEF_LIST
#include "generated/DialectTypeDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "generated/DialectOps.cpp.inc"
      >();
}

}  // namespace axon
