module;

#include "dialect/dialect.h"
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
import :codegen_inst;

namespace axon {

static auto get_inputs_type(Module& module, mlir::OpBuilder& builder)
    -> mlir::TupleType {
  llvm::SmallVector<mlir::Type> types;
  auto* mlir_ctx = builder.getContext();
  auto& inputs = module.forward().inputs();
  auto element_type = builder.getF32Type();

  for (auto input_id : inputs.iter()) {
    auto& input_info = inputs.get(input_id);
    auto input_type =
        mlir::RankedTensorType::get(input_info.shape, element_type);
    types.push_back(input_type);
  }

  return mlir::TupleType::get(mlir_ctx, types);
}

static auto get_cached_values_type(Module& module, mlir::OpBuilder& builder)
    -> mlir::TupleType {
  llvm::SmallVector<mlir::Type> types;
  auto* mlir_ctx = builder.getContext();
  auto element_type = builder.getF32Type();
  auto& forward = module.forward();

  for (auto inst_id : forward.cached_values().keys()) {
    auto shape = forward.get_shape(inst_id);
    auto input_type = mlir::RankedTensorType::get(shape, element_type);
    types.push_back(input_type);
  }
  return mlir::TupleType::get(mlir_ctx, types);
}

static auto codegen_module(Module& module, mlir::OpBuilder& builder,
                           mlir::ModuleOp& module_op) -> void {
  auto inputs_type = get_inputs_type(module, builder);
  auto cached_values_type = get_cached_values_type(module, builder);
  auto types = llvm::SmallVector<mlir::Type>{inputs_type, cached_values_type};

  codegen_graph(module.forward(), builder, module_op, "forward", types);
}

export auto codegen(Module& module, mlir::MLIRContext& mlir_ctx)
    -> mlir::ModuleOp {
  mlir_ctx.loadDialect<mlir::func::FuncDialect>();
  mlir_ctx.loadDialect<mlir::tensor::TensorDialect>();
  mlir_ctx.loadDialect<mlir::linalg::LinalgDialect>();
  mlir_ctx.loadDialect<AxonDialect>();

  mlir::OpBuilder builder{&mlir_ctx};
  auto module_op = mlir::ModuleOp::create(builder.getUnknownLoc());

  codegen_module(module, builder, module_op);

  return module_op;
}

}  // namespace axon
