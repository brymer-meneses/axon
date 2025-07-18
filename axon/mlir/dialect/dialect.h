#pragma once

#include <cstdint>
#include <print>
#include <tuple>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/HashBuilder.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect
#include "generated/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "generated/DialectTypeDefs.h.inc"

#define GET_OP_CLASSES
#include "generated/DialectOps.h.inc"
