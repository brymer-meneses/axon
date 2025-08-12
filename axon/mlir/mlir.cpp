module;

#include "dialect/dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

export module axon.mlir;

import axon.core;

export import :compilation_context;
export import :codegen_graph;

import :lowering;

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

  return module_op;
}

auto createLowerToLlvmPipeline(mlir::PassManager& manager) -> void {
  manager.addPass(axon::createStandardLoweringPass());
  manager.addPass(mlir::createCanonicalizerPass());

  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // manager.addPass(axon::createLlvmLoweringPass());
  // manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  // manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  // manager.addPass(mlir::createConvertFuncToLLVMPass());
  // manager.addPass(mlir::createCanonicalizerPass());
}

}  // namespace axon
