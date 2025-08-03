module;

#include "axon/base/dcheck.h"
#include "dialect/dialect.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

export module axon.mlir:codegen_inst;

import axon.core;
import axon.base;

import :compilation_context;

static constexpr auto CachedValuesIndex = 0;
static constexpr auto InputsIndex = 1;
static constexpr auto ForwardInputsIndex = 2;

namespace axon {

static auto get_inputs_type(Graph& graph, mlir::OpBuilder& builder)
    -> mlir::TupleType {
  llvm::SmallVector<mlir::Type> types;
  auto* mlir_ctx = builder.getContext();
  auto& inputs = graph.inputs();
  auto element_type = builder.getF32Type();

  for (auto input_id : inputs.iter()) {
    auto& input_info = inputs.get(input_id);
    auto input_type = mlir::MemRefType::get(input_info.shape, element_type);
    types.push_back(input_type);
  }

  return mlir::TupleType::get(mlir_ctx, types);
}

static auto get_argument(CompilationContext& ctx, int32_t argument_index,
                         uint64_t tuple_index) -> mlir::Value {
  auto* block = ctx.builder.getInsertionBlock();
  auto argument = block->getArgument(argument_index);
  auto tuple = llvm::dyn_cast<mlir::TupleType>(argument.getType());
  auto result_type = tuple.getType(tuple_index);
  auto accessed = ctx.builder.create<TupleAccessOp>(
      ctx.builder.getUnknownLoc(), result_type, argument, tuple_index);
  return {accessed};
}

static auto codegen(insts::InitialGradient op, CompilationContext& ctx)
    -> mlir::Value {
  return {};
}

static auto codegen(insts::SetCachedValue op, CompilationContext& ctx)
    -> mlir::Value {
  return {};
}

static auto codegen(insts::GetCachedValue op, CompilationContext& ctx)
    -> mlir::Value {
  AXON_DCHECK(op.cached_value_id.has_value(), "Invalid index");
  auto loc = ctx.builder.getUnknownLoc();
  auto memref =
      get_argument(ctx, CachedValuesIndex, op.cached_value_id.value());
  auto memref_type = llvm::dyn_cast<mlir::MemRefType>(memref.getType());
  auto tensor_type = mlir::RankedTensorType::get(memref_type.getShape(),
                                                 memref_type.getElementType());
  auto as_tensor = ctx.builder.create<mlir::bufferization::ToTensorOp>(
      loc, tensor_type, memref);
  return {as_tensor};
}

static auto codegen(insts::AccumulateGrad op, CompilationContext& ctx)
    -> mlir::Value {
  return {};
}

static auto codegen(insts::GetInput op, CompilationContext& ctx)
    -> mlir::Value {
  AXON_DCHECK(op.input_id.has_value(), "Invalid index");
  auto loc = ctx.builder.getUnknownLoc();
  auto memref = get_argument(ctx, InputsIndex, op.input_id.value());
  auto memref_type = llvm::dyn_cast<mlir::MemRefType>(memref.getType());
  auto tensor_type = mlir::RankedTensorType::get(memref_type.getShape(),
                                                 memref_type.getElementType());
  auto as_tensor = ctx.builder.create<mlir::bufferization::ToTensorOp>(
      loc, tensor_type, memref);
  return {as_tensor};
}

static auto codegen(insts::Return op, CompilationContext& ctx) -> mlir::Value {
  auto loc = ctx.builder.getUnknownLoc();
  auto returned = ctx.values[op.returned_id];
  ctx.builder.create<mlir::func::ReturnOp>(loc, returned);
  return {};
}

static auto codegen(insts::LocalTensor op, CompilationContext& ctx)
    -> mlir::Value {
  return {};
}

static auto codegen(insts::Add op, CompilationContext& ctx) -> mlir::Value {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  return ctx.builder.create<mlir::arith::AddFOp>(ctx.builder.getUnknownLoc(),
                                                 lhs, rhs);
}

static auto codegen(insts::Mul op, CompilationContext& ctx) -> mlir::Value {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  return ctx.builder.create<mlir::arith::MulFOp>(ctx.builder.getUnknownLoc(),
                                                 lhs, rhs);
}

export auto codegen_graph(Graph& graph, mlir::OpBuilder& builder,
                          mlir::ModuleOp& module_op, llvm::StringLiteral name,
                          llvm::SmallVector<mlir::Type> additional_args,
                          bool is_backward) -> void {
  CompilationContext ctx(graph, builder);

  additional_args.emplace_back(get_inputs_type(graph, builder));

  builder.setInsertionPointToEnd(module_op.getBody());
  auto loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type> return_types;

  if (not is_backward) {
    return_types.emplace_back(
        mlir::RankedTensorType::get({2, 3}, builder.getF32Type()));
  }
  auto func_type = builder.getFunctionType(additional_args, return_types);

  auto func_op = builder.create<mlir::func::FuncOp>(loc, name, func_type);
  auto* block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(block);

  for (auto inst_id : graph.insts().iter()) {
    const auto& inst = graph.insts().get(inst_id);
    inst.visit(
        [&](auto inst) { ctx.values.insert({inst_id, codegen(inst, ctx)}); });
  }
}

}  // namespace axon
