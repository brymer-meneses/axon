module;

#include <print>

#include "dialect/dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

export module axon.mlir;

export import :context;
export import :forward;

export namespace axon {

auto codegen(Module& module) -> mlir::ModuleOp {
  Context context{module};

  codegen_forward(context);
  return context.mlir_module();
}

}  // namespace axon
