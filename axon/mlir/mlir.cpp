module;

#include <print>

#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

export module axon.mlir;

export import :context;
export import :forward;

export namespace axon {

auto codegen(mlir::MLIRContext& ctx, Module& module) -> mlir::ModuleOp {
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::tensor::TensorDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<AxonDialect>();

  Context context{module, ctx};

  codegen_forward(context);
  return context.result();
}

}  // namespace axon
