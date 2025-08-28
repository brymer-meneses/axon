#include "dialect.h"

#include "generated/dialect.cpp.inc"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

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
