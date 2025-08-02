module;

#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

export module axon.mlir:codegen_module;

import axon.core;
import axon.base;

import :compilation_context;
import :codegen_inst;

namespace axon {

static auto get_inputs_type(CompilationContext& ctx) -> TensorRefListType {
  llvm::SmallVector<TensorRefType> types;
  auto* mlir_ctx = ctx.builder().getContext();
  auto& inputs = ctx.module().forward().inputs();
  auto element_type = ctx.builder().getF32Type();

  for (InputId input_id : inputs.iter()) {
    auto& input_info = inputs.get(input_id);
    auto input_type = TensorRefType::get(
        mlir_ctx, element_type, input_info.shape, input_info.requires_grad);
    types.push_back(input_type);
  }

  return TensorRefListType::get(mlir_ctx, types);
}

static auto get_cached_values_type(CompilationContext& ctx)
    -> TensorRefListType {
  llvm::SmallVector<TensorRefType> types;
  auto* mlir_ctx = ctx.builder().getContext();
  auto element_type = ctx.builder().getF32Type();
  auto& forward = ctx.module().forward();

  for (auto inst_id : forward.cached_values().keys()) {
    auto shape = forward.get_shape(inst_id);
    auto requires_grad = ctx.module().check_requires_grad(inst_id);
    auto input_type =
        TensorRefType::get(mlir_ctx, element_type, shape, requires_grad);
    types.push_back(input_type);
  }
  return TensorRefListType::get(mlir_ctx, types);
}

static auto codegen_module(CompilationContext& ctx) -> void {
  ctx.builder().setInsertionPointToEnd(ctx.module_op().getBody());

  auto loc = ctx.builder().getUnknownLoc();

  llvm::SmallVector<mlir::Type> args_type;

  args_type.emplace_back(get_inputs_type(ctx));
  args_type.emplace_back(get_cached_values_type(ctx));

  auto func_type = ctx.builder().getFunctionType(args_type, {});
  auto func =
      ctx.builder().create<mlir::func::FuncOp>(loc, "forward", func_type);
  ctx.builder().setInsertionPointToStart(func.addEntryBlock());
  ctx.builder().create<mlir::func::ReturnOp>(loc);
}

export auto codegen(Module& module, mlir::MLIRContext& mlir_ctx)
    -> mlir::ModuleOp {
  mlir_ctx.loadDialect<mlir::func::FuncDialect>();
  mlir_ctx.loadDialect<mlir::tensor::TensorDialect>();
  mlir_ctx.loadDialect<mlir::arith::ArithDialect>();
  mlir_ctx.loadDialect<AxonDialect>();

  CompilationContext compilation_ctx{module, mlir_ctx};
  codegen_module(compilation_ctx);

  return compilation_ctx.module_op();
}

}  // namespace axon
