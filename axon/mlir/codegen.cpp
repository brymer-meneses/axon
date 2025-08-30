module;
#include <ranges>

#include "axon/base/dcheck.h"
#include "dialect/dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

// MLIR Imports
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

export module axon.mlir:codegen;

import axon.core;
import axon.base;

namespace axon {

struct CompilationContext {
  Graph& graph;
  mlir::OpBuilder& builder;
  mlir::func::FuncOp& func_op;

  llvm::DenseMap<InstId, mlir::Value> tensor_refs{};
  llvm::DenseMap<InstId, mlir::Value> values{};
};

static auto computeExpandReassociation(mlir::TensorType lhs,
                                       mlir::TensorType rhs)
    -> llvm::SmallVector<mlir::ReassociationIndices> {
  llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation;

  int64_t lhs_rank = lhs.getRank();
  int64_t rhs_rank = rhs.getRank();

  assert(rhsRank >= lhsRank && "expand requires rhs to have higher rank");

  // How many extra leading dims rhs has
  int64_t extra = rhs_rank - lhs_rank;

  // First group: [0..extra] (extra leading dims + first matching one)
  {
    mlir::ReassociationIndices group;
    for (int64_t i = 0; i <= extra; i++) {
      group.push_back(i);
    }
    reassociation.emplace_back(std::move(group));
  }

  // Remaining groups: one-to-one mapping
  for (int64_t i = extra + 1; i < rhs_rank; i++) {
    reassociation.push_back({i});
  }

  return reassociation;
}

static auto codegen(insts::AccumulateGrad op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto from = ctx.tensor_refs[op.inst_id];
  auto value = ctx.values[op.value_id];

  AccumulateGradOp::create(ctx.builder, ctx.builder.getUnknownLoc(), from,
                           value);
}

static auto codegen(insts::Constant op, CompilationContext& ctx, InstId inst_id)
    -> void {
  auto& constant = ctx.graph.constants().get(op.constant_id);
  auto result_type =
      mlir::RankedTensorType::get(constant.shape(), ctx.builder.getF32Type());

  if (constant.element_type() == ElementType::Float32) {
    auto data_ptr = reinterpret_cast<float*>(constant.data_ptr());
    auto data_ref = llvm::ArrayRef<float>(data_ptr, constant.size());
    auto data_attribute = mlir::DenseElementsAttr::get(result_type, data_ref);

    ctx.values[inst_id] = ConstantOp::create(
        ctx.builder, ctx.builder.getUnknownLoc(), data_attribute);
  }
}

static auto codegen(insts::GetParameter op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto tensor_ref = ctx.func_op.getArgument(op.param_id.value());

  ctx.tensor_refs[inst_id] = tensor_ref;
  ctx.values[inst_id] =
      GetDataOp::create(ctx.builder, ctx.builder.getUnknownLoc(), tensor_ref);
}

static auto codegen(insts::OnesLike op, CompilationContext& ctx, InstId inst_id)
    -> void {
  auto like = ctx.values[op.inst_id];
  ctx.values[inst_id] =
      FillLikeOp::create(ctx.builder, ctx.builder.getUnknownLoc(), like, 1.0);
}

static auto normalizeTensorOperands(mlir::OpBuilder& builder, mlir::Value lhs,
                                    mlir::Value rhs)
    -> std::tuple<mlir::Value, mlir::Value, mlir::Type> {
  auto loc = builder.getUnknownLoc();

  auto lhs_tensor = mlir::cast<mlir::TensorType>(lhs.getType());
  auto rhs_tensor = mlir::cast<mlir::TensorType>(rhs.getType());
  auto lhs_rank = lhs_tensor.getRank();
  auto rhs_rank = lhs_tensor.getRank();

  auto result_type = lhs_rank >= rhs_rank ? lhs.getType() : rhs.getType();
  if (lhs_tensor.getRank() < rhs_tensor.getRank()) {
    auto reassociation_indices =
        computeExpandReassociation(lhs_tensor, rhs_tensor);

    lhs = mlir::tensor::ExpandShapeOp::create(builder, loc, result_type, lhs,
                                              reassociation_indices);
  } else if (lhs_tensor.getRank() > rhs_tensor.getRank()) {
    auto reassociation_indices =
        computeExpandReassociation(rhs_tensor, lhs_tensor);
    rhs = mlir::tensor::ExpandShapeOp::create(builder, loc, result_type, rhs,
                                              reassociation_indices);
  }
  return {lhs, rhs, result_type};
}

static auto codegen(insts::Add op, CompilationContext& ctx, InstId inst_id)
    -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  auto loc = ctx.builder.getUnknownLoc();

  mlir::Type result_type;
  std::tie(lhs, rhs, result_type) =
      normalizeTensorOperands(ctx.builder, lhs, rhs);

  ctx.values[inst_id] =
      mlir::tosa::AddOp::create(ctx.builder, loc, result_type, lhs, rhs);
}

static auto codegen(insts::Mul op, CompilationContext& ctx, InstId inst_id)
    -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  auto loc = ctx.builder.getUnknownLoc();

  mlir::Type result_type;
  std::tie(lhs, rhs, result_type) =
      normalizeTensorOperands(ctx.builder, lhs, rhs);

  auto shift_value_type =
      mlir::RankedTensorType::get({1}, ctx.builder.getI8Type());

  auto shift_value = mlir::tosa::ConstOp::create(
      ctx.builder, loc, shift_value_type,
      mlir::DenseElementsAttr::get(shift_value_type,
                                   ctx.builder.getI8IntegerAttr(0)));

  ctx.values[inst_id] = mlir::tosa::MulOp::create(ctx.builder, loc, result_type,
                                                  lhs, rhs, shift_value);
}

static auto codegen(insts::MatMul op, CompilationContext& ctx, InstId inst_id)
    -> void {
  // auto lhs = ctx.values[op.lhs_id];
  // auto rhs = ctx.values[op.rhs_id];
  // ctx.values[inst_id] =
  //     handleBinaryOp<mlir::tosa::MatMulOp>(ctx.builder, lhs, rhs);
}

static auto codegen(insts::Transpose op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  // auto lhs = ctx.values[op.lhs_id];
  // auto rhs = ctx.values[op.rhs_id];
  //
  // ctx.values[inst_id] =
  //     MulOp::create(ctx.builder, ctx.builder.getUnknownLoc(), lhs, rhs);
}

static auto getFunctionType(Graph& graph, mlir::OpBuilder& builder)
    -> mlir::FunctionType {
  llvm::SmallVector<mlir::Type> arg_types;
  auto element_type = builder.getF32Type();
  for (auto& param : graph.parameters().values()) {
    auto requires_grad = param.requires_grad;
    auto shape = graph.getShape(param.inst_id);
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

  mlir::func::ReturnOp::create(ctx.builder, loc);
}

}  // namespace axon
