module;

#include "dialect/dialect.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

export module axon.mlir;

import axon.core;

export import :codegen;

import :standard_lowering;
import :llvm_lowering;

export namespace axon {

auto createDialectRegistry() -> mlir::DialectRegistry {
  mlir::DialectRegistry registry;

  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<AxonDialect>();

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);

  return registry;
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
  manager.addPass(mlir::createCanonicalizerPass());

  if (opts.level == LoweringOps::Level::LLVM) {
    manager.addPass(mlir::createConvertElementwiseToLinalgPass());
    manager.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());

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
