module;

#include "axon/base/dcheck.h"
#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

export module axon.mlir:codegen_inst;

import axon.core;
import axon.base;

import :compilation_context;

static constexpr auto ParamsIndex = 0;
static constexpr auto CachedValuesIndex = 1;
static constexpr auto InputsIndex = 2;
static constexpr auto ForwardInputsIndex = 2;

namespace axon {

static auto get_argument(CompilationContext& ctx, int32_t argument_index,
                         uint64_t tuple_index) -> mlir::Value {
  auto* block = ctx.builder.getInsertionBlock();
  AXON_DCHECK(block != nullptr, "");
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
  return {};
}

static auto codegen(insts::AccumulateGrad op, CompilationContext& ctx)
    -> mlir::Value {
  return {};
}

static auto codegen(insts::GetInput op, CompilationContext& ctx)
    -> mlir::Value {
  AXON_DCHECK(op.input_id.has_value(), "Invalid index");
  return get_argument(ctx, 0, op.input_id.value());
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
                          mlir::ModuleOp& module_op, std::string_view name,
                          llvm::SmallVector<mlir::Type> arg_types) -> void {
  CompilationContext ctx(graph, builder);

  builder.setInsertionPointToEnd(module_op.getBody());
  auto loc = builder.getUnknownLoc();

  auto result_type = mlir::RankedTensorType::get({2, 3}, builder.getF32Type());
  auto func_type = builder.getFunctionType(arg_types, {result_type});

  auto func = builder.create<mlir::func::FuncOp>(loc, name, func_type);
  auto* block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  for (auto inst_id : graph.insts().iter()) {
    const auto& inst = graph.insts().get(inst_id);
    inst.visit(
        [&](auto inst) { ctx.values.insert({inst_id, codegen(inst, ctx)}); });
  }
}

}  // namespace axon
