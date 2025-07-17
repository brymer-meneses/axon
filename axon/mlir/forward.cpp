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

export module axon.mlir:forward;

import axon.core;
import axon.base;

import :context;

namespace axon {

auto get_input_list(Context& context) -> TensorRefListType {
  llvm::SmallVector<TensorRefType> inputs;
  auto* mlir_context = context.builder().getContext();
  auto& module = context.module();
  for (InstId tensor_id : module.input_tensors()) {
    const auto& tensor_data = module.forward_tensors().at(tensor_id);
    auto input_type = TensorRefType::get(
        mlir_context, context.builder().getF32Type(), tensor_data.shape,
        module.check_requires_grad(tensor_id));
    inputs.push_back(input_type);
  }
  return TensorRefListType::get(mlir_context, inputs);
}

// Codegen function signature.
auto codegen_function(Context& context) -> mlir::func::FuncOp {
  llvm::SmallVector<mlir::Type> input_types;

  input_types.push_back(get_input_list(context));
  auto unknown_loc = context.builder().getUnknownLoc();

  // The return type will be inferred later.
  auto func_type = context.builder().getFunctionType(input_types, {});
  auto func = context.builder().create<mlir::func::FuncOp>(
      unknown_loc, "forward", func_type);
  return func;
}

auto codegen_inst(Context& context, InstId inst_id, const Inst& inst) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto* block = context.builder().getInsertionBlock();
  auto& values = context.forward_values();

  if (auto get_input = inst.try_get_as<insts::GetInput>()) {
    auto input_list =
        llvm::dyn_cast<TensorRefListType>(block->getArgument(0).getType());
    auto index = get_input->input_id.value();
    auto result_type = input_list[index];
    auto tensor_ref = context.builder().create<ListAccessOp>(
        loc, result_type, block->getArgument(0), index);
    values[inst_id] = context.builder().create<GetDataOp>(loc, tensor_ref);
  }

  // if (auto local_tensor = inst.try_get_as<insts::LocalTensor>()) {
  //   const auto& tensor_data = context.module().forward_tensors().at(inst_id);
  //   const auto& shape = tensor_data.shape;
  //
  //   values[inst_id] = context.builder().create<mlir::RankedTensorType>(
  //       loc, shape, context.builder().getF32Type());
  // }

  if (auto add = inst.try_get_as<insts::Add>()) {
    auto lhs = values[add->lhs_id];
    auto rhs = values[add->rhs_id];
    auto result = context.builder().create<mlir::arith::AddFOp>(loc, lhs, rhs);
    values[inst_id] = result;
  }
}

export auto codegen_forward(Context& context) -> void {
  auto loc = context.builder().getUnknownLoc();
  auto func = codegen_function(context);
  context.builder().setInsertionPointToStart(func.addEntryBlock());

  llvm::DenseMap<InstId, mlir::Value> values;
  for (auto [inst_id, inst] : context.module().forward_insts().iter_values()) {
    codegen_inst(context, inst_id, inst);
  }

  context.builder().create<mlir::func::ReturnOp>(loc);
}

}  // namespace axon
