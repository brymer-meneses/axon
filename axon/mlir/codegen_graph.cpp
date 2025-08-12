module;

#include "axon/base/dcheck.h"
#include "dialect/dialect.h"
#include "llvm/ADT/StringRef.h"

// MLIR Imports
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

static auto codegen(insts::AccumulateGrad op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto from = ctx.tensor_refs[op.inst_id];
  auto value = ctx.values[op.value_id];

  ctx.builder.create<AccumulateGradOp>(ctx.builder.getUnknownLoc(), from,
                                       value);
}

static auto codegen(insts::Constant op, CompilationContext& ctx, InstId inst_id)
    -> void {
  auto& data = ctx.graph.data().get(op.data_id);
  auto result_type =
      mlir::RankedTensorType::get(data.shape, ctx.builder.getF32Type());

  auto data_ref = llvm::ArrayRef<float>(data.data);
  auto data_attribute = mlir::DenseElementsAttr::get(result_type, data_ref);

  ctx.values[inst_id] = ctx.builder.create<ConstantOp>(
      ctx.builder.getUnknownLoc(), data_attribute);
}

static auto codegen(insts::GetParameter op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto tensor_ref = ctx.func_op.getArgument(op.param_id.value());

  ctx.tensor_refs[inst_id] = tensor_ref;
  ctx.values[inst_id] =
      ctx.builder.create<GetDataOp>(ctx.builder.getUnknownLoc(), tensor_ref);
}

static auto codegen(insts::Add op, CompilationContext& ctx, InstId inst_id)
    -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];

  ctx.values[inst_id] =
      ctx.builder.create<AddOp>(ctx.builder.getUnknownLoc(), lhs, rhs);
}

static auto codegen(insts::Mul op, CompilationContext& ctx, InstId inst_id)
    -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];

  ctx.values[inst_id] =
      ctx.builder.create<MulOp>(ctx.builder.getUnknownLoc(), lhs, rhs);
}

static auto getFunctionType(Graph& graph, mlir::OpBuilder& builder)
    -> mlir::FunctionType {
  llvm::SmallVector<mlir::Type> arg_types;
  for (auto& param : graph.parameters().values()) {
    auto requires_grad = param.requires_grad;
    auto shape = graph.getShape(param.inst_id);
    auto element_type = builder.getF32Type();
    auto type = TensorRefType::get(builder.getContext(), element_type, shape,
                                   requires_grad);
    arg_types.emplace_back(type);
  }
  return builder.getFunctionType(arg_types, {});
}

export auto codegenGraph(Graph& graph, mlir::OpBuilder& builder,
                         mlir::ModuleOp& module_op) -> void {
  builder.setInsertionPointToEnd(module_op.getBody());
  auto loc = builder.getUnknownLoc();

  auto func_type = getFunctionType(graph, builder);
  auto func_op = builder.create<mlir::func::FuncOp>(loc, "graph", func_type);

  builder.setInsertionPointToStart(func_op.addEntryBlock());
  CompilationContext ctx(graph, builder, func_op);

  for (auto inst_id : graph.insts().keys()) {
    const auto& inst = graph.insts().get(inst_id);
    inst.visit([&](auto inst) { codegen(inst, ctx, inst_id); });
  }
  ctx.builder.create<mlir::func::ReturnOp>(loc);
}

}  // namespace axon
