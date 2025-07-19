module;

#include <print>

#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

export module axon.mlir:forward;

import axon.core;
import axon.base;

import :context;

namespace axon {

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

// Codegen function signature.
static auto codegen_function(Context& context) -> mlir::func::FuncOp {
  llvm::SmallVector<mlir::Type> args_type;

  args_type.push_back(inputs_type(context));
  args_type.push_back(cached_values_type(context));

  auto unknown_loc = context.builder().getUnknownLoc();

  // The return type will be inferred later.
  auto func_type = context.builder().getFunctionType(args_type, {});
  auto func = context.builder().create<mlir::func::FuncOp>(
      unknown_loc, "forward", func_type);

  return func;
}

static auto codegen_inst(Context& context, InstId inst_id) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto* block = context.builder().getInsertionBlock();
  auto& values = context.forward_values();
  const auto& inst = context.module().forward_insts().get(inst_id);

  if (auto get_input = inst.try_get_as<insts::GetInput>()) {
    auto input_list =
        llvm::dyn_cast<TensorRefListType>(block->getArgument(0).getType());
    auto index = get_input->input_id.value();
    auto result_type = input_list[index];
    auto tensor_ref = context.builder().create<ListAccessOp>(
        loc, result_type, block->getArgument(0), index);
    values[inst_id] = context.builder().create<GetDataOp>(loc, tensor_ref);
  }

  if (auto local_tensor = inst.try_get_as<insts::LocalTensor>()) {
    const auto& data = context.module().get_data(inst_id, /*is_forward=*/true);
    auto result_type = mlir::RankedTensorType::get(
        data.shape(), context.builder().getF32Type());
    auto data_attribute = mlir::DenseElementsAttr::get(result_type, data.ref());
    values[inst_id] = context.builder().create<ConstantOp>(loc, data_attribute);
  }

  if (auto add = inst.try_get_as<insts::Add>()) {
    auto lhs = values[add->lhs_id];
    auto rhs = values[add->rhs_id];
    values[inst_id] =
        context.builder().create<mlir::arith::AddFOp>(loc, lhs, rhs);
  }

  if (auto mul = inst.try_get_as<insts::Mul>()) {
    auto lhs = values[mul->lhs_id];
    auto rhs = values[mul->rhs_id];
    values[inst_id] =
        context.builder().create<mlir::arith::MulFOp>(loc, lhs, rhs);
  }
}

export auto codegen_forward(Context& context) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto func = codegen_function(context);
  context.builder().setInsertionPointToStart(func.addEntryBlock());

  for (auto inst_id : context.module().forward_insts().iter()) {
    codegen_inst(context, inst_id);
  }

  context.builder().create<mlir::func::ReturnOp>(loc);
}

}  // namespace axon
