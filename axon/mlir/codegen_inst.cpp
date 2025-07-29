module;

#include "axon/base/dcheck.h"
#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

export module axon.mlir:codegen_inst;

import axon.core;
import axon.base;

import :context;

namespace axon {

template <Index T>
static auto get_argument(Context& context, T id, bool is_forward)
    -> mlir::Value {
  auto& map = context.get_argument_map<T>(is_forward);
  if (map.contains(id)) {
    return map[id];
  }

  auto* block = context.builder().getInsertionBlock();
  auto loc = context.builder().getUnknownLoc();

  auto argument_index = context.get_argument_index<T>();
  auto argument = block->getArgument(argument_index);

  auto list = llvm::dyn_cast<TensorRefListType>(argument.getType());
  auto index = id.value();
  auto result_type = list[index];
  auto tensor_ref =
      context.builder().create<ListAccessOp>(loc, result_type, argument, index);
  map[id] = tensor_ref;
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

static auto codegen(Context& context, insts::SetCachedValue op, InstId,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto to_set = get_argument(context, op.cached_value_id, is_forward);
  auto tensor = context.tensors(is_forward)[op.new_value_id];
  AXON_DCHECK(tensor != nullptr, "");

  context.builder().create<SetDataOp>(loc, to_set, tensor);
}

static auto codegen(Context& context, insts::GetCachedValue op, InstId inst_id,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto& tensors = context.tensors(is_forward);
  auto tensor_ref = get_argument(context, op.cached_value_id, is_forward);
  tensors[inst_id] = context.builder().create<GetDataOp>(loc, tensor_ref);
}

static auto codegen(Context& context, insts::AccumulateGrad op, InstId,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();

  auto input_tensor_ref = get_argument(context, op.input_id, is_forward);
  auto tensor = context.tensors(is_forward)[op.value_id];

  context.builder().create<AccumulateGradOp>(loc, input_tensor_ref, tensor);
}

static auto codegen(Context& context, insts::GetInput op, InstId inst_id,
                    bool is_forward) -> void {
  auto loc = context.builder().getUnknownLoc();

  auto tensor_ref = get_argument(context, op.input_id, is_forward);
  auto& tensors = context.tensors(is_forward);

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
