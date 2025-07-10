#include "dialect.h"

#include "Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "DialectTypeDefs.cpp.inc"

#define GET_OP_CLASSES
#include "DialectOps.cpp.inc"
namespace axon {

auto AxonDialect::initialize() -> void {
  addTypes<
#define GET_TYPEDEF_LIST
#include "DialectTypeDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "DialectOps.cpp.inc"
      >();
}

}  // namespace axon
