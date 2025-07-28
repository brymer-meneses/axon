module;

#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

export module axon.mlir:codegen_inst;

import axon.core;
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

  return {tensor_ref};
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

export auto codegen_inst(Context& context, InstId inst_id, bool is_forward)
    -> void {
  auto& inst = context.module().get_inst(inst_id, is_forward);

  inst.visit([&](const auto op) { codegen(context, op, inst_id, is_forward); });
}

}  // namespace axon
