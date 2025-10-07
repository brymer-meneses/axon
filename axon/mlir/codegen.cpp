module;

#include "axon/base/macros.h"
#include "dialect/dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/OpDefinition.h"

// MLIR Imports
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

  InstId current_id;
};

static auto getElementType(DataType data_type, mlir::OpBuilder& builder)
    -> mlir::Type {
  switch (data_type.kind()) {
    case DataType::Float32:
      return builder.getF32Type();
    case DataType::Float64:
      return builder.getF64Type();
    case DataType::Int1:
      return builder.getI1Type();
    case DataType::Int32:
      return builder.getI32Type();
    case DataType::Int64:
      return builder.getI64Type();
  }

  AXON_UNREACHABLE("Unsupported data type");
}

static auto getFloatAttr(mlir::Type element_type, f64 value)
    -> mlir::FloatAttr {
  return mlir::FloatAttr::get(element_type, value);
}

static auto codegen(const insts::AccumulateGrad& op, CompilationContext& ctx)
    -> void {
  auto source = ctx.values[op.source_id];
  auto param_id = ctx.inst_to_param.getValueOf(op.sink_id);
  AXON_ASSERT(param_id.isValid());

  auto grad_memref = ctx.grad_memrefs[param_id];
  auto sink_tensor_type = mlir::cast<mlir::RankedTensorType>(source.getType());
  auto sink = mlir::bufferization::ToTensorOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), sink_tensor_type, grad_memref,
      /*restrict=*/true, /*writable=*/true);
  AccumulateOp::create(ctx.builder, ctx.builder.getUnknownLoc(), sink, source);
}

static auto codegen(const insts::AccumulateData& op, CompilationContext& ctx)
    -> void {
  auto source = ctx.values[op.source_id];
  auto param_id = ctx.inst_to_param.getValueOf(op.sink_id);
  AXON_ASSERT(param_id.isValid());

  auto data_memref = ctx.data_memrefs[param_id];
  auto sink_tensor_type = mlir::cast<mlir::RankedTensorType>(source.getType());
  auto sink = mlir::bufferization::ToTensorOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), sink_tensor_type, data_memref,
      /*restrict=*/true, /*writable=*/true);
  AccumulateOp::create(ctx.builder, ctx.builder.getUnknownLoc(), sink, source);
}

static auto codegen(const insts::Constant& op, CompilationContext& ctx)
    -> void {
  auto inst_id = ctx.current_id;

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
    case DataType::Int1: {
      auto data_ptr = reinterpret_cast<bool*>(constant->data_ptr());
      llvm::SmallVector<bool> values(data_ptr, data_ptr + constant->size());
      data_attribute = mlir::DenseElementsAttr::get(result_type, values);
      break;
    }
    case DataType::Int32: {
      auto data_ptr = reinterpret_cast<i32*>(constant->data_ptr());
      auto data_ref = llvm::ArrayRef<i32>(data_ptr, constant->size());
      data_attribute = mlir::DenseElementsAttr::get(result_type, data_ref);
      break;
    }
    case DataType::Int64: {
      auto data_ptr = reinterpret_cast<i64*>(constant->data_ptr());
      auto data_ref = llvm::ArrayRef<i64>(data_ptr, constant->size());
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
static auto codegenReduceInst(const InstType& op, CompilationContext& ctx)
    -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] =
      LoweredOpType::create(ctx.builder, ctx.builder.getUnknownLoc(),
                            result_type, input, op.axis, op.keep_dims);
}

static auto codegen(const insts::Sum& op, CompilationContext& ctx) -> void {
  codegenReduceInst<insts::Sum, SumOp>(op, ctx);
}

static auto codegen(const insts::Mean& op, CompilationContext& ctx) -> void {
  codegenReduceInst<insts::Mean, MeanOp>(op, ctx);
}

static auto codegen(const insts::ArgMax& op, CompilationContext& ctx) -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];
  auto shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(shape, ctx.builder.getI64Type());

  ctx.values[inst_id] =
      ArgMaxOp::create(ctx.builder, ctx.builder.getUnknownLoc(), result_type,
                       input, op.axis, op.keep_dims);
}

template <typename InstType, typename LoweredOp>
static auto codegenElementWiseBinaryInst(const InstType& op,
                                         CompilationContext& ctx) -> void {
  auto inst_id = ctx.current_id;
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  ctx.values[inst_id] =
      LoweredOp::create(ctx.builder, ctx.builder.getUnknownLoc(), lhs, rhs);
}

static auto codegen(const insts::Add& op, CompilationContext& ctx) -> void {
  codegenElementWiseBinaryInst<insts::Add, AddOp>(op, ctx);
}

static auto codegen(const insts::Sub& op, CompilationContext& ctx) -> void {
  codegenElementWiseBinaryInst<insts::Sub, SubOp>(op, ctx);
};

static auto codegen(const insts::Mul& op, CompilationContext& ctx) -> void {
  codegenElementWiseBinaryInst<insts::Mul, MulOp>(op, ctx);
}

static auto codegen(const insts::Softmax& op, CompilationContext& ctx) -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] = SoftmaxOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, input, op.axis);
}

static auto codegen(const insts::ExpandDims& op, CompilationContext& ctx)
    -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  llvm::SmallVector<mlir::Attribute> expansions;
  llvm::SmallVector<i64> result_shape{tensor_type.getShape()};

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

static auto codegen(const insts::GetParameter& op, CompilationContext& ctx)
    -> void {
  auto inst_id = ctx.current_id;
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

static auto codegen(const insts::FillLike& op, CompilationContext& ctx)
    -> void {
  auto like = ctx.values[op.input_id];
  auto like_type = mlir::cast<mlir::RankedTensorType>(like.getType());
  auto element_type = like_type.getElementType();

  mlir::Attribute fill_attr;
  switch (op.fill_value.data_type().kind()) {
    case DataType::Float32:
      fill_attr = getFloatAttr(element_type, op.fill_value.as<f32>());
      break;
    case DataType::Float64:
      fill_attr = getFloatAttr(element_type, op.fill_value.as<f64>());
      break;
    case DataType::Int1:
      fill_attr = ctx.builder.getBoolAttr(op.fill_value.as<bool>());
      break;
    case DataType::Int32:
      fill_attr =
          ctx.builder.getIntegerAttr(element_type, op.fill_value.as<i32>());
      break;
    case DataType::Int64:
      fill_attr =
          ctx.builder.getIntegerAttr(element_type, op.fill_value.as<i64>());
      break;
  }

  auto inst_id = ctx.current_id;
  ctx.values[inst_id] = FillOp::create(ctx.builder, ctx.builder.getUnknownLoc(),
                                       like.getType(), fill_attr);
}

static auto codegen(const insts::MatMul& op, CompilationContext& ctx) -> void {
  auto inst_id = ctx.current_id;
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];

  auto data_type =
      mlir::cast<mlir::RankedTensorType>(lhs.getType()).getElementType();
  auto result_type =
      mlir::RankedTensorType::get(ctx.graph.getShape(inst_id), data_type);

  ctx.values[inst_id] = MatMulOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, lhs, rhs);
}

static auto codegen(const insts::Transpose& op, CompilationContext& ctx)
    -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];
  auto result_shape = ctx.graph.getShape(inst_id);
  auto data_type =
      mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
  auto result_type = mlir::RankedTensorType::get(result_shape, data_type);

  ctx.values[inst_id] =
      TransposeOp::create(ctx.builder, ctx.builder.getUnknownLoc(), result_type,
                          input, op.from, op.to);
}

static auto codegen(const insts::Squeeze& op, CompilationContext& ctx) -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] = SqueezeOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, input, op.dim);
}

static auto codegen(const insts::Unsqueeze& op, CompilationContext& ctx)
    -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_shape = ctx.graph.getShape(inst_id);
  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  ctx.values[inst_id] = UnsqueezeOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), result_type, input, op.dim);
}

static auto codegen(const insts::Reshape& op, CompilationContext& ctx) -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];
  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());

  auto result_type = mlir::RankedTensorType::get(op.target_shape,
                                                 tensor_type.getElementType());

  ctx.values[inst_id] =
      ReshapeOp::create(ctx.builder, ctx.builder.getUnknownLoc(), result_type,
                        input, op.target_shape);
}

static auto codegen(const insts::ScalarMul& op, CompilationContext& ctx)
    -> void {
  auto input = ctx.values[op.input_id];
  auto data_type = op.scalar.data_type();
  auto element_type =
      mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();

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
    case DataType::Int1: {
      attr = ctx.builder.getBoolAttr(op.scalar.as<bool>());
      break;
    }
    case DataType::Int32: {
      attr = ctx.builder.getIntegerAttr(element_type, op.scalar.as<i32>());
      break;
    }
    case DataType::Int64: {
      attr = ctx.builder.getIntegerAttr(element_type, op.scalar.as<i64>());
      break;
    }
  }

  auto inst_id = ctx.current_id;
  ctx.values[inst_id] = ScalarMulOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), input, attr);
};

static auto codegen(const insts::Pow& op, CompilationContext& ctx) -> void {
  auto input = ctx.values[op.input_id];
  auto inst_id = ctx.current_id;

  auto exponent_value = op.exponent.as<f32>();

  ctx.values[inst_id] =
      PowOp::create(ctx.builder, ctx.builder.getUnknownLoc(), input,
                    ctx.builder.getF64FloatAttr(exponent_value));
}

static auto codegen(const insts::Relu& op, CompilationContext& ctx) -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];

  ctx.values[inst_id] =
      ReluOp::create(ctx.builder, ctx.builder.getUnknownLoc(), input);
}

static auto getComparisonPredicate(mlir::MLIRContext* context,
                                   insts::Compare::Predicate predicate) {
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
}

static auto codegen(const insts::Compare& op, CompilationContext& ctx) -> void {
  auto lhs = ctx.values[op.lhs_id];
  auto rhs = ctx.values[op.rhs_id];
  auto inst_id = ctx.current_id;

  auto context = ctx.builder.getContext();
  auto predicate = getComparisonPredicate(context, op.predicate);

  ctx.values[inst_id] = CompareOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), lhs, rhs, predicate);
}

static auto codegen(const insts::Neg& op, CompilationContext& ctx) -> void {
  auto input = ctx.values[op.input_id];
  auto inst_id = ctx.current_id;
  ctx.values[inst_id] =
      NegOp::create(ctx.builder, ctx.builder.getUnknownLoc(), input);
};

static auto codegen(const insts::Log& op, CompilationContext& ctx) -> void {
  auto inst_id = ctx.current_id;
  auto input = ctx.values[op.input_id];
  ctx.values[inst_id] = mlir::math::LogOp::create(
      ctx.builder, ctx.builder.getUnknownLoc(), input);
};

static auto getFunctionType(const Graph& graph,
                            llvm::ArrayRef<InstId> returned_ids,
                            mlir::OpBuilder& builder) -> mlir::FunctionType {
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

  llvm::SmallVector<mlir::Type> returns;
  for (auto returned_id : returned_ids) {
    auto shape = graph.getShape(returned_id);
    auto returned_dtype = graph.getDataType(returned_id);
    auto returned_element_type = getElementType(returned_dtype, builder);
    auto returned_type =
        mlir::RankedTensorType::get(shape, returned_element_type);
    returns.push_back(returned_type);
  }

  return builder.getFunctionType(arg_types, returns);
}

export auto codegenGraph(const Graph& graph,
                         llvm::ArrayRef<InstId> returned_ids,
                         mlir::OpBuilder& builder) -> mlir::ModuleOp {
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto loc = builder.getUnknownLoc();

  auto func_type = getFunctionType(graph, returned_ids, builder);
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
    ctx.current_id = inst_id;
    const auto& inst = graph.insts().get(inst_id);
    inst.visit([&](const auto& op) { codegen(op, ctx); });
  }

  llvm::SmallVector<mlir::Value> returns;
  for (auto returned_id : returned_ids) {
    returns.push_back(ctx.values[returned_id]);
  }

  mlir::func::ReturnOp::create(ctx.builder, loc, returns);
  return module;
}

}  // namespace axon
