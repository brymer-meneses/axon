module;

#include "axon/base/macros.h"
#include "dialect/dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
  const Graph& graph;
  mlir::OpBuilder& builder;
  mlir::func::FuncOp& func_op;
  llvm::DenseMap<InstId, mlir::Value> values{};

  // Mapping from ParamId to function argument memrefs
  llvm::DenseMap<ParamId, mlir::Value> data_memrefs{};
  llvm::DenseMap<ParamId, mlir::Value> grad_memrefs{};

  RelationalStore<InstId, ParamId> inst_to_param;
};

static auto getElementType(DataType data_type, mlir::OpBuilder& builder)
    -> mlir::Type {
  switch (data_type.kind()) {
    case DataType::Float32:
      return builder.getF32Type();
    case DataType::Float64:
      return builder.getF64Type();
  }

  AXON_UNREACHABLE("Unsupported data type");
}

static auto getFloatAttr(mlir::Type element_type, f64 value)
    -> mlir::FloatAttr {
  return mlir::FloatAttr::get(element_type, value);
}

static auto codegen(const insts::AccumulateGrad& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto source = ctx.values[op.source_id];
  auto param_id = ctx.inst_to_param.getValueOf(op.sink_id);

  auto grad_memref = ctx.grad_memrefs[param_id];
  auto sink_tensor_type = mlir::cast<mlir::RankedTensorType>(source.getType());
  auto sink = mlir::bufferization::ToTensorOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), sink_tensor_type, grad_memref,
      /*restrict=*/true, /*writable=*/true);
  AccumulateOp::create(ctx.builder, ctx.builder.getUnknownLoc(), sink, source);
}

static auto codegen(const insts::AccumulateData& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto source = ctx.values[op.source_id];
  auto param_id = ctx.inst_to_param.getValueOf(op.sink_id);

  auto data_memref = ctx.data_memrefs[param_id];
  auto sink_tensor_type = mlir::cast<mlir::RankedTensorType>(source.getType());
  auto sink = mlir::bufferization::ToTensorOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), sink_tensor_type, data_memref,
      /*restrict=*/true, /*writable=*/true);
  AccumulateOp::create(ctx.builder, ctx.builder.getUnknownLoc(), sink, source);
}

static auto codegen(const insts::Constant& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  Storage* constant = ctx.graph.constants().get(inst_id)->get();
  AXON_ASSERT(constant != nullptr);

  auto element_type = getElementType(constant->data_type(), ctx.builder);
  auto result_type =
      mlir::RankedTensorType::get(constant->shape(), element_type);

  mlir::DenseElementsAttr data_attribute;
  switch (constant->data_type().kind()) {
    case DataType::Float32: {
      auto data_ptr = reinterpret_cast<f32*>(constant->data_ptr());
      auto data_ref = llvm::ArrayRef<f32>(data_ptr, constant->size());
      data_attribute = mlir::DenseElementsAttr::get(result_type, data_ref);
      break;
    }
    case DataType::Float64: {
      auto data_ptr = reinterpret_cast<f64*>(constant->data_ptr());
      auto data_ref = llvm::ArrayRef<f64>(data_ptr, constant->size());
      data_attribute = mlir::DenseElementsAttr::get(result_type, data_ref);
      break;
    }
  }

  ctx.values[inst_id] = ConstantOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), data_attribute);
}

/// Utility function to handle lowering of reduce like inst which all have the
/// same fields.
template <typename InstType, typename LoweredOpType>
static auto codegenReduceInst(const InstType& op, CompilationContext& ctx,
                              InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] =
      LoweredOpType::create(ctx.builder, ctx.builder.getUnknownLoc(),
                            result_type, input, op.axis, op.keep_dims);
}

static auto codegen(const insts::Sum& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  codegenReduceInst<insts::Sum, SumOp>(op, ctx, inst_id);
}

static auto codegen(const insts::Mean& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  codegenReduceInst<insts::Mean, MeanOp>(op, ctx, inst_id);
}

static auto codegen(const insts::Softmax& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] = SoftmaxOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, input, op.axis);
}

static auto codegen(const insts::ExpandDims& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

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
                           result_type, input, expansions_attr);
}

static auto codegen(const insts::GetParameter& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto data_memref = ctx.data_memrefs[op.param_id];
  ctx.inst_to_param.createRelation(inst_id, op.param_id);

  auto shape = ctx.graph.getShape(inst_id);
  auto element_type =
      getElementType(ctx.graph.getDataType(inst_id), ctx.builder);
  auto tensor_type = mlir::RankedTensorType::get(shape, element_type);
  ctx.values[inst_id] = mlir::bufferization::ToTensorOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), tensor_type, data_memref,
      /*restrict=*/true, /*writable=*/false);
}

static auto codegen(const insts::FillLike& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto like = ctx.values[op.input_id];
  // Extract the scalar value according to its stored dtype.
  double fill;
  switch (op.fill_value.data_type().kind()) {
    case DataType::Float32: {
      fill = static_cast<double>(op.fill_value.as<f32>());
      break;
    }
    case DataType::Float64: {
      fill = static_cast<double>(op.fill_value.as<f64>());
      break;
    }
  }
  auto like_type = mlir::cast<mlir::RankedTensorType>(like.getType());
  auto element_type = like_type.getElementType();

  ctx.values[inst_id] =
      FillOp::create(ctx.builder, ctx.builder.getUnknownLoc(), like.getType(),
                     getFloatAttr(element_type, fill));
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
  auto input = ctx.values[op.input_id];
  auto result_shape = ctx.graph.getShape(inst_id);
  auto data_type =
      mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
  auto result_type = mlir::RankedTensorType::get(result_shape, data_type);

  ctx.values[inst_id] =
      TransposeOp::create(ctx.builder, ctx.builder.getUnknownLoc(), result_type,
                          input, op.from, op.to);
}

static auto codegen(const insts::Squeeze& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] = SqueezeOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, input, op.dim);
}

static auto codegen(const insts::Unsqueeze& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] = UnsqueezeOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, input, op.dim);
}

static auto codegen(const insts::Reshape& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_type = mlir::RankedTensorType::get(op.target_shape,
                                                 tensor_type.getElementType());

  ctx.values[inst_id] =
      ReshapeOp::create(ctx.builder, ctx.builder.getUnknownLoc(), result_type,
                        input, op.target_shape);
}

static auto codegen(const insts::ScalarMul& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];
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
      ctx.builder, ctx.builder.getUnknownLoc(), input, attr);
};

static auto codegen(const insts::Pow& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];

  auto exponent_value = op.exponent.as<f32>();

  ctx.values[inst_id] =
      PowOp::create(ctx.builder, ctx.builder.getUnknownLoc(), input,
                    ctx.builder.getF64FloatAttr(exponent_value));
}

static auto codegen(const insts::Relu& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];

  ctx.values[inst_id] =
      ReluOp::create(ctx.builder, ctx.builder.getUnknownLoc(), input);
}

static auto codegen(const insts::Compare& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  static constexpr auto comparison_predicate =
      [](mlir::MLIRContext* context, insts::Compare::Predicate predicate) {
        switch (predicate) {
          case insts::Compare::Predicate::Equal:
            return ComparePredicateAttr::get(context, ComparePredicate::eq);
          case insts::Compare::Predicate::Less:
            return ComparePredicateAttr::get(context, ComparePredicate::lt);
          case insts::Compare::Predicate::LessEq:
            return ComparePredicateAttr::get(context, ComparePredicate::le);
          case insts::Compare::Predicate::Greater:
            return ComparePredicateAttr::get(context, ComparePredicate::gt);
          case insts::Compare::Predicate::GreaterEq:
            return ComparePredicateAttr::get(context, ComparePredicate::ge);
          case insts::Compare::Predicate::NotEqual:
            return ComparePredicateAttr::get(context, ComparePredicate::ne);
        }
      };

  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  auto context = ctx.builder.getContext();
  auto predicate = comparison_predicate(context, op.predicate);

  ctx.values[inst_id] = CompareOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), lhs, rhs, predicate);
}

static auto codegen(const insts::Sub& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];

  ctx.values[inst_id] =
      SubOp::create(ctx.builder, ctx.builder.getUnknownLoc(), lhs, rhs);
};

static auto codegen(const insts::Neg& op, CompilationContext& ctx,
                    InstId inst_id) -> void {
  auto input = ctx.values[op.input_id];

  ctx.values[inst_id] =
      NegOp::create(ctx.builder, ctx.builder.getUnknownLoc(), input);
};

static auto getFunctionType(const Graph& graph, mlir::OpBuilder& builder)
    -> mlir::FunctionType {
  llvm::SmallVector<mlir::Type> arg_types;
  auto context = builder.getContext();
  AXON_ASSERT(context != nullptr);

  for (auto [param_id, param] : graph.parameters().keysAndValues()) {
    auto shape = graph.getShape(param.inst_id);
    auto element_type = getElementType(param.data_type, builder);
    auto memref_type = mlir::MemRefType::get(shape, element_type);
    arg_types.emplace_back(memref_type);
    if (param.requires_grad) {
      arg_types.emplace_back(memref_type);
    }
  }

  auto returned_id = graph.getReturnedId();
  if (!returned_id) {
    return builder.getFunctionType(arg_types, {});
  }

  auto shape = graph.getShape(returned_id);
  auto returned_dtype = graph.getDataType(returned_id);
  auto returned_element_type = getElementType(returned_dtype, builder);
  auto returned_type =
      mlir::RankedTensorType::get(shape, returned_element_type);
  return builder.getFunctionType(arg_types, {returned_type});
}

export auto codegenGraph(const Graph& graph, mlir::OpBuilder& builder)
    -> mlir::ModuleOp {
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto loc = builder.getUnknownLoc();

  auto func_type = getFunctionType(graph, builder);
  auto func_op = builder.create<mlir::func::FuncOp>(loc, "graph", func_type);

  builder.setInsertionPointToStart(func_op.addEntryBlock());
  CompilationContext ctx(graph, builder, func_op);

  auto arg_index = 0;
  for (auto [param_id, param] : graph.parameters().keysAndValues()) {
    auto data_arg = func_op.getArgument(arg_index);
    ctx.data_memrefs[param_id] = data_arg;
    arg_index += 1;

    if (param.requires_grad) {
      auto grad_arg = func_op.getArgument(arg_index);
      ctx.grad_memrefs[param_id] = grad_arg;
      arg_index += 1;
    }
  }

  for (auto inst_id : graph.insts().keys()) {
    const auto& inst = graph.insts().get(inst_id);
    inst.visit([&](const auto& op) { codegen(op, ctx, inst_id); });
  }

  if (auto returned_id = graph.getReturnedId()) {
    mlir::func::ReturnOp::create(ctx.builder, loc, ctx.values[returned_id]);
  } else {
    mlir::func::ReturnOp::create(ctx.builder, loc);
  }

  return module;
}

}  // namespace axon
