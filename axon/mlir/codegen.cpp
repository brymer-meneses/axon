module;

#include "axon/base/macros.h"
#include "dialect/dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/OpDefinition.h"

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

static auto codegen(const insts::AccumulateGrad& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto tensor_ref = ctx.tensor_refs[op.inst_id];
  auto tensor_ref_type = mlir::cast<TensorRefType>(tensor_ref.getType());

  auto source = ctx.values[op.value_id];

  auto sink = GetGradOp::create(ctx.builder, ctx.builder.getUnknownLoc(),
                                tensor_ref_type.getTensorType(), tensor_ref);

  AccumulateOp::create(ctx.builder, ctx.builder.getUnknownLoc(), sink, source);
}

static auto codegen(const insts::Constant& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  const Storage& constant = ctx.graph.constants().get(inst_id)->get();

  auto result_type =
      mlir::RankedTensorType::get(constant.shape(), ctx.builder.getF32Type());

  if (constant.data_type() == DataType::Float32) {
    auto data_ptr = reinterpret_cast<float*>(constant.data_ptr());
    auto data_ref = llvm::ArrayRef<float>(data_ptr, constant.size());
    auto data_attribute = mlir::DenseElementsAttr::get(result_type, data_ref);

    ctx.values[inst_id] = ConstantOp::create(
        ctx.builder, ctx.builder.getUnknownLoc(), data_attribute);
  }
}

static auto codegen(const insts::Sum& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto operand = ctx.values[op.operand_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(operand.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] =
      SumOp::create(ctx.builder, ctx.builder.getUnknownLoc(), result_type,
                    operand, op.axis, op.keepdims);
}

static auto codegen(const insts::ExpandDims& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto operand = ctx.values[op.operand_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(operand.getType());

  llvm::SmallVector<mlir::Attribute> expansions;
  llvm::SmallVector<int64_t> result_shape{tensor_type.getShape()};

  // Create an array attribute for each (dim, scale) pair
  for (auto mapping : op.mappings) {
    llvm::SmallVector<mlir::Attribute, 2> pair = {
        ctx.builder.getI64IntegerAttr(mapping.dim),
        ctx.builder.getI64IntegerAttr(mapping.scale),
    };
    expansions.push_back(ctx.builder.getArrayAttr(pair));
    result_shape[mapping.dim] = mapping.scale;
  }

  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());
  auto expansions_attr = ctx.builder.getArrayAttr(expansions);

  ctx.values[inst_id] =
      ExpandDimsOp::create(ctx.builder, ctx.builder.getUnknownLoc(),
                           result_type, operand, expansions_attr);
}

static auto codegen(const insts::GetParameter& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto tensor_ref = ctx.func_op.getArgument(op.param_id.value());

  ctx.tensor_refs[inst_id] = tensor_ref;
  ctx.values[inst_id] =
      GetDataOp::create(ctx.builder, ctx.builder.getUnknownLoc(), tensor_ref);
}

static auto codegen(const insts::OnesLike& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto like = ctx.values[op.operand_id];

  ctx.values[inst_id] =
      FillOp::create(ctx.builder, ctx.builder.getUnknownLoc(), like.getType(),
                     ctx.builder.getF32FloatAttr(1.0));
}

static auto codegen(const insts::Add& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  ctx.values[inst_id] =
      AddOp::create(ctx.builder, ctx.builder.getUnknownLoc(), lhs, rhs);
}

static auto codegen(const insts::Mul& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];

  ctx.values[inst_id] =
      MulOp::create(ctx.builder, ctx.builder.getUnknownLoc(), lhs, rhs);
}

static auto codegen(const insts::MatMul& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];

  auto data_type =
      mlir::cast<mlir::RankedTensorType>(lhs.getType()).getElementType();
  auto result_type =
      mlir::RankedTensorType::get(ctx.graph.getShape(inst_id), data_type);

  ctx.values[inst_id] = MatMulOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, lhs, rhs);
}

static auto codegen(const insts::Transpose& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto operand = ctx.values[op.operand_id];
  auto result_shape = ctx.graph.getShape(inst_id);
  auto data_type =
      mlir::cast<mlir::RankedTensorType>(operand.getType()).getElementType();
  auto result_type = mlir::RankedTensorType::get(result_shape, data_type);

  ctx.values[inst_id] =
      TransposeOp::create(ctx.builder, ctx.builder.getUnknownLoc(), result_type,
                          operand, op.from, op.to);
}

static auto codegen(const insts::Squeeze& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto operand = ctx.values[op.operand_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(operand.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] = SqueezeOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, operand, op.dim);
}

static auto codegen(const insts::Unsqueeze& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto operand = ctx.values[op.operand_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(operand.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] = UnsqueezeOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, operand, op.dim);
}

static auto codegen(const insts::Reshape& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto operand = ctx.values[op.operand_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(operand.getType());

  auto result_type = mlir::RankedTensorType::get(op.target_shape,
                                                 tensor_type.getElementType());

  ctx.values[inst_id] =
      ReshapeOp::create(ctx.builder, ctx.builder.getUnknownLoc(), result_type,
                        operand, op.target_shape);
}

static auto codegen(const insts::ScalarMul& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto operand = ctx.values[op.operand_id];
  auto data_type = op.scalar.data_type();

  mlir::Attribute attr;
  switch (data_type.kind()) {
    case DataType::Float32: {
      attr = ctx.builder.getF32FloatAttr(op.scalar.as<f32>());
      break;
    }
    case DataType::Float64: {
      attr = ctx.builder.getF64FloatAttr(op.scalar.as<f64>());
      break;
    }
  }

  ctx.values[inst_id] = ScalarMulOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), operand, attr);
};

static auto codegen(const insts::Sub& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];

  ctx.values[inst_id] =
      SubOp::create(ctx.builder, ctx.builder.getUnknownLoc(), lhs, rhs);
};

static auto codegen(const insts::Neg& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto operand = ctx.values[op.operand_id];

  ctx.values[inst_id] =
      NegOp::create(ctx.builder, ctx.builder.getUnknownLoc(), operand);
};

static auto getFunctionType(Graph& graph, mlir::OpBuilder& builder)
    -> mlir::FunctionType {
  llvm::SmallVector<mlir::Type> arg_types;
  auto data_type = builder.getF32Type();
  for (auto& param : graph.parameters().values()) {
    auto requires_grad = param.requires_grad;
    auto shape = graph.getShape(param.inst_id);
    auto type = TensorRefType::get(builder.getContext(), data_type, shape,
                                   requires_grad);
    arg_types.emplace_back(type);
  }

  auto returned_id = graph.getReturnedId();
  if (!returned_id) {
    return builder.getFunctionType(arg_types, {});
  }

  auto shape = graph.getShape(returned_id);
  auto returned_type = mlir::RankedTensorType::get(shape, data_type);
  return builder.getFunctionType(arg_types, {returned_type});
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
    inst.visit([&](const auto& inst) { codegen(inst, ctx, inst_id); });
  }

  if (auto returned_id = graph.getReturnedId()) {
    mlir::func::ReturnOp::create(ctx.builder, loc, ctx.values[returned_id]);
  } else {
    mlir::func::ReturnOp::create(ctx.builder, loc);
  }
}

}  // namespace axon
