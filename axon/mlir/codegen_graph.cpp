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

export module axon.mlir:codegen_graph;

import axon.core;
import axon.base;

import :compilation_context;

namespace axon {

// `ArgumentIndex` is a helper to easily declare the argument index of a
// function in the module. The forward pass contains the following arguments:
//  - cached values
//  - inputs
//  - buffers
//    - this is only present if the graph is a backward graph.
//  These three arguments are passed as a tuple containing pointers to the
//  buffer for these tensors.
template <typename IndexType>
struct ArgumentIndex;

template <>
struct ArgumentIndex<CachedValueId> {
  static constexpr auto value = 0;
};

template <>
struct ArgumentIndex<InputId> {
  static constexpr auto value = 1;
};

template <>
struct ArgumentIndex<BufferId> {
  static constexpr auto value = 2;
};

template <Index IndexType>
static auto get_argument(CompilationContext& ctx, IndexType index)
    -> mlir::Value {
  auto func_args_size = ctx.func_op.getArgumentTypes().size();
  auto argument_index = ArgumentIndex<IndexType>::value;

  AXON_DCHECK(static_cast<int32_t>(func_args_size) > argument_index, "{} < {}",
              func_args_size, argument_index);

  auto argument = ctx.func_op.getArgument(argument_index);
  auto tuple = llvm::dyn_cast<mlir::TupleType>(argument.getType());
  auto tuple_index = index.value();
  auto result_type = tuple.getType(tuple_index);

  if (auto lookup = ctx.args.lookup({argument_index, tuple_index})) {
    return lookup;
  }

  auto accessed = ctx.builder.create<TupleAccessOp>(
      ctx.builder.getUnknownLoc(), result_type, argument, tuple_index);
  return accessed;
}

static auto codegen(insts::SetCachedValue op, CompilationContext& ctx)
    -> mlir::Value {
  auto cached_value = get_argument(ctx, op.cached_value_id);
  auto to_store = ctx.values[op.new_value_id];
  ctx.builder.create<StoreOp>(ctx.builder.getUnknownLoc(), cached_value,
                              to_store);
  return {};
}

static auto codegen(insts::GetCachedValue op, CompilationContext& ctx)
    -> mlir::Value {
  AXON_DCHECK(op.cached_value_id.has_value(), "Invalid index");
  auto loc = ctx.builder.getUnknownLoc();
  auto memref = get_argument(ctx, op.cached_value_id);
  auto memref_type = llvm::dyn_cast<mlir::MemRefType>(memref.getType());
  auto tensor_type = mlir::RankedTensorType::get(memref_type.getShape(),
                                                 memref_type.getElementType());
  auto as_tensor = ctx.builder.create<mlir::bufferization::ToTensorOp>(
      loc, tensor_type, memref);
  return as_tensor;
}

static auto codegen(insts::AccumulateGrad op, CompilationContext& ctx)
    -> mlir::Value {
  auto value = ctx.values[op.value_id];
  auto buffer = get_argument(ctx, op.buffer_id);

  ctx.builder.create<AccumulateOp>(ctx.builder.getUnknownLoc(), buffer, value);
  return {};
}

static auto codegen(insts::GetInput op, CompilationContext& ctx)
    -> mlir::Value {
  AXON_DCHECK(op.input_id.has_value(), "Invalid index");
  auto loc = ctx.builder.getUnknownLoc();
  auto memref = get_argument(ctx, op.input_id);
  auto memref_type = llvm::dyn_cast<mlir::MemRefType>(memref.getType());
  auto tensor_type = mlir::RankedTensorType::get(memref_type.getShape(),
                                                 memref_type.getElementType());
  auto as_tensor = ctx.builder.create<mlir::bufferization::ToTensorOp>(
      loc, tensor_type, memref);
  return as_tensor;
}

static auto codegen(insts::Return op, CompilationContext& ctx) -> mlir::Value {
  auto loc = ctx.builder.getUnknownLoc();
  if (op.returned_id.has_value()) {
    auto returned = ctx.values[op.returned_id];
    ctx.builder.create<mlir::func::ReturnOp>(loc, returned);
  } else {
    ctx.builder.create<mlir::func::ReturnOp>(loc);
  }
  return {};
}

static auto codegen(insts::LocalTensor op, CompilationContext& ctx)
    -> mlir::Value {
  return {};
}

static auto codegen(insts::Add op, CompilationContext& ctx) -> mlir::Value {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  return {ctx.builder.create<AddOp>(ctx.builder.getUnknownLoc(), lhs, rhs)};
}

static auto codegen(insts::Mul op, CompilationContext& ctx) -> mlir::Value {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  return {ctx.builder.create<MulOp>(ctx.builder.getUnknownLoc(), lhs, rhs)};
}

export auto codegen_graph(Graph& graph, mlir::OpBuilder& builder,
                          mlir::ModuleOp& module_op, llvm::StringLiteral name,
                          llvm::SmallVector<mlir::Type> additional_args,
                          bool is_backward) -> void {
  builder.setInsertionPointToEnd(module_op.getBody());
  auto loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type> return_types;

  if (not is_backward) {
    return_types.emplace_back(
        mlir::RankedTensorType::get({2, 3}, builder.getF32Type()));
  }
  auto func_type = builder.getFunctionType(additional_args, return_types);

  auto func_op = builder.create<mlir::func::FuncOp>(loc, name, func_type);
  builder.setInsertionPointToStart(func_op.addEntryBlock());

  CompilationContext ctx(graph, builder, func_op);

  for (auto inst_id : graph.insts().iter()) {
    const auto& inst = graph.insts().get(inst_id);
    inst.visit(
        [&](auto inst) { ctx.values.insert({inst_id, codegen(inst, ctx)}); });
  }
}

}  // namespace axon
