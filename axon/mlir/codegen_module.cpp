module;

#include "dialect/dialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

export module axon.mlir:codegen_module;

import axon.core;
import axon.base;

import :compilation_context;
import :codegen_graph;

namespace axon {

static auto get_cached_values_type(Graph& graph, mlir::OpBuilder& builder)
    -> mlir::TupleType {
  llvm::SmallVector<mlir::Type> types;
  auto* mlir_ctx = builder.getContext();
  auto element_type = builder.getF32Type();
  for (auto inst_id : graph.cached_values().keys()) {
    auto shape = graph.get_shape(inst_id);
    auto input_type = mlir::MemRefType::get(shape, element_type);
    types.push_back(input_type);
  }
  return mlir::TupleType::get(mlir_ctx, types);
}

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

static auto codegen_module(Module& module, mlir::OpBuilder& builder,
                           mlir::ModuleOp& module_op) -> void {
  llvm::SmallVector<mlir::Type> arg_types;
  arg_types.emplace_back(get_cached_values_type(module.forward(), builder));
  arg_types.emplace_back(get_inputs_type(module.forward(), builder));

  codegen_graph(module.forward(), builder, module_op, "forward", arg_types,
                /*is_backward=*/false);

  arg_types.clear();

  arg_types.emplace_back(get_cached_values_type(module.forward(), builder));
  arg_types.emplace_back(get_inputs_type(module.backward(), builder));
  arg_types.emplace_back(get_inputs_type(module.forward(), builder));

  codegen_graph(module.backward(), builder, module_op, "backward", arg_types,
                /*is_backward=*/true);
}

export auto codegen(Module& module, mlir::MLIRContext& mlir_ctx)
    -> mlir::ModuleOp {
  mlir_ctx.loadDialect<mlir::func::FuncDialect>();
  mlir_ctx.loadDialect<mlir::tensor::TensorDialect>();
  mlir_ctx.loadDialect<mlir::linalg::LinalgDialect>();
  mlir_ctx.loadDialect<mlir::bufferization::BufferizationDialect>();
  mlir_ctx.loadDialect<AxonDialect>();

  mlir::OpBuilder builder{&mlir_ctx};
  auto module_op = mlir::ModuleOp::create(builder.getUnknownLoc());

  codegen_module(module, builder, module_op);

  return module_op;
}

}  // namespace axon
