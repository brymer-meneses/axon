module;

#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

export module axon.mlir;

import axon.core;

export import :compilation_context;
export import :codegen_graph;

export namespace axon {

auto codegen(Graph& graph, mlir::MLIRContext& mlir_ctx) -> mlir::ModuleOp {
  mlir_ctx.loadDialect<mlir::func::FuncDialect>();
  mlir_ctx.loadDialect<mlir::tensor::TensorDialect>();
  mlir_ctx.loadDialect<mlir::linalg::LinalgDialect>();
  mlir_ctx.loadDialect<mlir::bufferization::BufferizationDialect>();
  mlir_ctx.loadDialect<AxonDialect>();

  mlir::OpBuilder builder{&mlir_ctx};
  auto module_op = mlir::ModuleOp::create(builder.getUnknownLoc());

  codegenGraph(graph, builder, module_op);

  // mlir::PassManager pm(&mlir_ctx);
  //
  // pm.addPass(createAxonLoweringPass());
  //
  // auto result = pm.run(module_op);
  // if (result.failed()) {
  //   return {};
  // }
  //
  return module_op;
}

}  // namespace axon
