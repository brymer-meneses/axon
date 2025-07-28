module;

#include "axon/base/dcheck.h"
#include "dialect/dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

export module axon.mlir:codegen;

import axon.core;
import axon.base;

import :context;

namespace axon {

static auto get_input(Context& context, InputId input_id, bool is_forward)
    -> mlir::Value {
  auto& inputs = context.inputs(is_forward);
  if (inputs.contains(input_id)) {
    return inputs[input_id];
  }

  auto* block = context.builder().getInsertionBlock();
  auto loc = context.builder().getUnknownLoc();

  auto input_list =
      llvm::dyn_cast<TensorRefListType>(block->getArgument(0).getType());

  auto index = input_id.value();
  auto result_type = input_list[index];

  auto tensor_ref = context.builder().create<ListAccessOp>(
      loc, result_type, block->getArgument(0), index);

  return tensor_ref;
}

static auto codegen(Context& context, insts::InitialGradient, InstId inst_id,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto* block = context.builder().getInsertionBlock();

  auto& tensors = context.tensors(is_forward);

  tensors[inst_id] =
      context.builder().create<GetDataOp>(loc, block->getArgument(2));
}

static auto codegen(Context&, insts::SetCachedValue, InstId, bool) -> void {
  // auto loc = context.builder().getUnknownLoc();
  // auto* block = context.builder().getInsertionBlock();
  // auto cached_values =
  //     llvm::dyn_cast<TensorRefListType>(block->getArgument(1).getType());
  //
  // auto& tensors = context.tensors(is_forward);
  //
  // auto index = op.cached_value_id.value();
  // auto result_type = cached_values[index];
  // auto tensor_ref = context.builder().create<ListAccessOp>(
  //     loc, result_type, block->getArgument(1), index);
  //
  // tensors[inst_id] = context.builder().create<GetDataOp>(loc, tensor_ref);
}

static auto codegen(Context& context, insts::GetCachedValue op, InstId inst_id,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto* block = context.builder().getInsertionBlock();
  auto cached_values =
      llvm::dyn_cast<TensorRefListType>(block->getArgument(1).getType());

  auto& tensors = context.tensors(is_forward);

  auto index = op.cached_value_id.value();
  auto result_type = cached_values[index];
  auto tensor_ref = context.builder().create<ListAccessOp>(
      loc, result_type, block->getArgument(1), index);

  tensors[inst_id] = context.builder().create<GetDataOp>(loc, tensor_ref);
}

static auto codegen(Context& context, insts::AccumulateGrad op, InstId,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();

  auto input_tensor_ref = get_input(context, op.input_id, is_forward);
  auto tensor = context.tensors(is_forward)[op.value_id];

  context.builder().create<AccumulateGradOp>(loc, input_tensor_ref, tensor);
}

static auto codegen(Context& context, insts::GetInput op, InstId inst_id,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();

  auto tensor_ref = get_input(context, op.input_id, is_forward);
  auto& inputs = context.inputs(is_forward);
  auto& tensors = context.tensors(is_forward);

  inputs[op.input_id] = tensor_ref;
  tensors[inst_id] = context.builder().create<GetDataOp>(loc, tensor_ref);
}

static auto codegen(Context& context, insts::LocalTensor, InstId inst_id,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto& tensors = context.tensors(is_forward);

  const auto& data = context.module().get_data(inst_id, is_forward);
  auto result_type =
      mlir::RankedTensorType::get(data.shape(), context.builder().getF32Type());
  auto data_attribute = mlir::DenseElementsAttr::get(result_type, data.ref());

  tensors[inst_id] = context.builder().create<ConstantOp>(loc, data_attribute);
}

static auto codegen(Context& context, insts::Add op, InstId inst_id,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto& tensors = context.tensors(is_forward);

  auto lhs = tensors[op.lhs_id];
  auto rhs = tensors[op.rhs_id];
  tensors[inst_id] =
      context.builder().create<mlir::arith::AddFOp>(loc, lhs, rhs);
}

static auto codegen(Context& context, insts::Mul op, InstId inst_id,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto& tensors = context.tensors(is_forward);

  auto lhs = tensors[op.lhs_id];
  auto rhs = tensors[op.rhs_id];
  tensors[inst_id] =
      context.builder().create<mlir::arith::MulFOp>(loc, lhs, rhs);
}

static auto inputs_type(Context& context) -> TensorRefListType {
  llvm::SmallVector<TensorRefType> inputs;
  auto* mlir_context = context.builder().getContext();
  auto& module = context.module();
  for (InstId tensor_id : module.input_tensors()) {
    const auto& data = module.get_data(tensor_id, /*is_forward=*/true);
    auto input_type =
        TensorRefType::get(mlir_context, context.builder().getF32Type(),
                           data.shape(), module.check_requires_grad(tensor_id));
    inputs.push_back(input_type);
  }
  return TensorRefListType::get(mlir_context, inputs);
}

static auto cached_values_type(Context& context) -> TensorRefListType {
  llvm::SmallVector<TensorRefType> cached_values;
  auto* mlir_context = context.builder().getContext();
  auto& module = context.module();

  for (InstId tensor_id : module.cached_values().keys()) {
    const auto& data = module.get_data(tensor_id, /*is_forward=*/true);
    auto inst_type =
        TensorRefType::get(mlir_context, context.builder().getF32Type(),
                           data.shape(), module.check_requires_grad(tensor_id));
    cached_values.push_back(inst_type);
  }
  return TensorRefListType::get(mlir_context, cached_values);
}

static auto codegen_function_proto(Context& context, std::string_view name,
                                   bool is_forward) -> mlir::func::FuncOp {
  llvm::SmallVector<mlir::Type> args_type;
  auto* mlir_context = context.builder().getContext();

  args_type.emplace_back(inputs_type(context));
  args_type.emplace_back(cached_values_type(context));

  if (not is_forward) {
    // Hardcode for now, the return type should be inferred from module.
    args_type.emplace_back(
        TensorRefType::get(mlir_context, context.builder().getF32Type(),
                           /*shape=*/{2, 3}, /*requires_grad=*/true));
  }

  auto unknown_loc = context.builder().getUnknownLoc();

  // The return type will be inferred later.
  auto func_type = context.builder().getFunctionType(args_type, {});
  auto func = context.builder().create<mlir::func::FuncOp>(unknown_loc, name,
                                                           func_type);

  return func;
}

static auto codegen_inst(Context& context, InstId inst_id, bool is_forward)
    -> void {
  auto& inst = context.module().get_inst(inst_id, is_forward);

  inst.visit([&](const auto op) { codegen(context, op, inst_id, is_forward); });
}

static auto codegen_forward(Context& context) -> void {
  context.builder().setInsertionPointToEnd(context.mlir_module().getBody());

  auto loc = context.builder().getUnknownLoc();
  auto func = codegen_function_proto(context, "forward", /*is_forward=*/true);
  context.builder().setInsertionPointToStart(func.addEntryBlock());

  for (auto inst_id : context.module().forward_insts().iter()) {
    codegen_inst(context, inst_id, /*is_forward=*/true);
  }

  context.builder().create<mlir::func::ReturnOp>(loc);
}

static auto codegen_backward(Context& context) -> void {
  context.builder().setInsertionPointToEnd(context.mlir_module().getBody());

  auto loc = context.builder().getUnknownLoc();
  auto func = codegen_function_proto(context, "backward", /*is_forward=*/false);
  context.builder().setInsertionPointToStart(func.addEntryBlock());

  for (auto inst_id : context.module().backward_insts().iter()) {
    codegen_inst(context, inst_id, /*is_forward=*/false);
  }

  context.builder().create<mlir::func::ReturnOp>(loc);
}

export auto codegen(mlir::MLIRContext& ctx, Module& module) -> mlir::ModuleOp {
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::tensor::TensorDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<AxonDialect>();

  Context context{module, ctx};

  codegen_forward(context);
  codegen_backward(context);

  return context.mlir_module();
}

}  // namespace axon
