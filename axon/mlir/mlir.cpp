module;

#include "dialect/dialect.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
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
  registry.insert<AxonDialect>();

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);

  return registry;
}

enum class LoweringLevel {
  Axon,
  Standard,
  Linalg,
  Bufferization,
  Affine,
  LLVM,
};

auto buildLlvmLoweringPipeline(mlir::PassManager& manager, LoweringLevel level)
    -> void {
  if (level == LoweringLevel::Axon) {
    return;
  }

  if (level >= LoweringLevel::Standard) {
    manager.addPass(axon::createStandardLoweringPass());
    manager.addPass(mlir::createCanonicalizerPass());
    manager.addPass(mlir::createCSEPass());
  }

  if (level >= LoweringLevel::Linalg) {
    manager.addPass(mlir::createConvertElementwiseToLinalgPass());
    manager.addPass(mlir::createLinalgElementwiseOpFusionPass());
    manager.addPass(mlir::createLinalgFoldIntoElementwisePass());

    manager.addNestedPass<mlir::func::FuncOp>(
        mlir::bufferization::createEmptyTensorEliminationPass());
  }

  if (level >= LoweringLevel::Bufferization) {
    mlir::bufferization::OneShotBufferizePassOptions bufferization_options;
    manager.addPass(
        mlir::bufferization::createOneShotBufferizePass(bufferization_options));
    manager.addPass(mlir::createCanonicalizerPass());
    // manager.addPass(mlir::createCSEPass());

    // Now cleanup memrefs.
    manager.addPass(mlir::memref::createExpandStridedMetadataPass());
    // manager.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
    // manager.addPass(
    //     mlir::bufferization::createDropEquivalentBufferResultsPass());

    // Allocation optimizations: hoist + promote *before* lowering dealloc.
    manager.addNestedPass<mlir::func::FuncOp>(
        mlir::bufferization::createBufferHoistingPass());
    manager.addNestedPass<mlir::func::FuncOp>(
        mlir::bufferization::createBufferLoopHoistingPass());
    // manager.addNestedPass<mlir::func::FuncOp>(
    //     mlir::bufferization::createPromoteBuffersToStackPass());

    // Now lower deallocs.
    manager.addNestedPass<mlir::func::FuncOp>(
        mlir::bufferization::createLowerDeallocationsPass());
    manager.addPass(
        mlir::bufferization::createBufferDeallocationSimplificationPass());
    mlir::bufferization::BufferDeallocationPipelineOptions deallocation_options;
    mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                         deallocation_options);

    manager.addPass(mlir::createCanonicalizerPass());
    // manager.addPass(mlir::createCSEPass());
  }

  if (level >= LoweringLevel::Affine) {
    manager.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    manager.addPass(mlir::createCanonicalizerPass());
    manager.addPass(mlir::affine::createLoopFusionPass());
  }

  if (level >= LoweringLevel::LLVM) {
    manager.addPass(axon::createLlvmLoweringPass());
    manager.addPass(mlir::createReconcileUnrealizedCastsPass());
    manager.addPass(mlir::createCanonicalizerPass());
  }
}

}  // namespace axon
