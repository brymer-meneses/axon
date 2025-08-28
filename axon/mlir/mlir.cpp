module;

#include "dialect/dialect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

export module axon.mlir;

import axon.core;

export import :codegen_graph;

export import :lowering;

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

struct LoweringOps {
  enum class Level {
    Axon,
    Standard,
    LLVM,
  };

  Level level;
};

auto buildLlvmLoweringPipeline(mlir::PassManager& manager, LoweringOps opts)
    -> void {
  if (opts.level == LoweringOps::Level::Axon) {
    return;
  }

  manager.addPass(axon::createStandardLoweringPass());

  if (opts.level == LoweringOps::Level::LLVM) {
    manager.addPass(mlir::createConvertElementwiseToLinalgPass());

    mlir::bufferization::OneShotBufferizePassOptions bufferization_options;
    manager.addPass(
        mlir::bufferization::createOneShotBufferizePass(bufferization_options));

    mlir::bufferization::BufferDeallocationPipelineOptions deallocation_options;
    mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                         deallocation_options);

    manager.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    manager.addPass(mlir::affine::createLoopFusionPass());

    manager.addPass(axon::createLlvmLoweringPass());

    manager.addPass(mlir::createCanonicalizerPass());
  }
}

}  // namespace axon
