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

import :context;

namespace axon::backend {

static auto inputs_type(Context& ctx) -> TensorRefListType {
  llvm::SmallVector<TensorRefType> inputs;

  auto* mlir_context = ctx.builder.getContext();
  for (auto tensor_id : ctx.module.input_tensors) {
    auto& data = module.get_data(tensor_id, /*is_forward=*/true);
    auto input_type = TensorRefType::get(
        mlir_context, ctx.builder.getF32Type(), data.shape,
        ctx.module.forward_function.check_requires_grad(tensor_id));
    inputs.push_back(input_type);
  }
  return TensorRefListType::get(mlir_context, inputs);
}

static auto cached_values_type(Context& ctx) -> TensorRefListType {
  llvm::SmallVector<TensorRefType> cached_values;
  auto* mlir_context = context.builder.getContext();

  for (auto tensor_id : module.cached_values.keys()) {
    auto& data = module.get_data(tensor_id, /*is_forward=*/true);
    auto inst_type = TensorRefType::get(
        mlir_context, ctx.builder.getF32Type(), data.shape(),
        ctx.module.forward_function.check_requires_grad(tensor_id));
    cached_values.push_back(inst_type);
  }
  return TensorRefListType::get(mlir_context, cached_values);
}
//
static auto codegen_function_proto(Context& context, std::string_view name,
                                   bool is_forward) -> mlir::func::FuncOp {
  llvm::SmallVector<mlir::Type> args_type;
  auto* mlir_context = ctx.builder.getContext();

  args_type.emplace_back(inputs_type(context));
  args_type.emplace_back(cached_values_type(context));

  if (not is_forward) {
    // Hardcode for now, the return type should be inferred from module.
    args_type.emplace_back(
        TensorRefType::get(mlir_context, ctx.builder.getF32Type(),
                           /*shape=*/{2, 3}, /*requires_grad=*/true));
  }

  auto unknown_loc = ctx.builder.getUnknownLoc();

  // The return type will be inferred later.
  auto func_type = ctx.builder.getFunctionType(args_type, {});
  auto func =
      ctx.builder.create<mlir::func::FuncOp>(unknown_loc, name, func_type);

  return func;
}

static auto codegen_forward(Context& context) -> void {
  context.builder.setInsertionPointToEnd(ctx.mlir_module.getBody());

  auto loc = context.builder.getUnknownLoc();
  auto func = codegen_function_proto(ctx, "forward", /*is_forward=*/true);
  ctx.builder.setInsertionPointToStart(func.addEntryBlock());

  // for (auto inst_id : context.module().forward_insts().iter()) {
  //   codegen_inst(context, inst_id, /*is_forward=*/true);
  // }

  ctx.builder.create<mlir::func::ReturnOp>(loc);
}

static auto codegen_backward(Context& context) -> void {
  ctx.builder.setInsertionPointToEnd(ctx.mlir_module.getBody());

  auto loc = ctx.builder.getUnknownLoc();
  auto func = codegen_function_proto(ctx, "backward", /*is_forward=*/false);
  ctx.builder.setInsertionPointToStart(func.addEntryBlock());

  // for (auto inst_id : context.module().backward_insts().iter()) {
  //   codegen_inst(context, inst_id, /*is_forward=*/false);
  // }

  ctx.builder.create<mlir::func::ReturnOp>(loc);
}

export auto codegen(mlir::MLIRContext& ctx, core::Module& module)
    -> mlir::ModuleOp {
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::tensor::TensorDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<AxonDialect>();

  Context context{module, ctx};

  codegen_forward(context);
  codegen_backward(context);

  return context.mlir_module;
}

}  // namespace axon::backend
