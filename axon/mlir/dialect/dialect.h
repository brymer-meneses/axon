#pragma once

#include <cstdint>
#include <print>
#include <tuple>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/HashBuilder.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect
#include "generated/dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "generated/dialect_type_defs.h.inc"

#define GET_OP_CLASSES
#include "generated/dialect_ops.h.inc"
