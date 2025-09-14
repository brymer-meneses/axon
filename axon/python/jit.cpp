module;

#include <memory>
#include <print>
#include <ranges>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

export module axon.python:jit;

import axon.base;
import axon.core;
import axon.mlir;

import :tensor;
import :abi;

namespace axon {

class CompiledFunction {
 public:
  CompiledFunction(mlir::MLIRContext* mlir_context)
      : mlir_context_(mlir_context) {}

  ~CompiledFunction() {
    for (auto descriptor : descriptors_) {
      auto tensor_descriptor =
          reinterpret_cast<abi::TensorDescriptor*>(descriptor);
      abi::TensorDescriptor::destroy(tensor_descriptor);
    }

    engine_.reset();
  }

  auto execute(GraphContext& context) -> void {
    // context contains a graph that is isomorphic to the one that was used to
    // construct this CompiledFunction.

    configureDescriptors(context);
    if (!engine_) {
      compileModuleAndPrepareEngine(context);
    }

    auto invocation_result = engine_->invokePacked("graph", descriptors_);
    if (invocation_result) {
      throw std::runtime_error("JIT invocation failed\n");
    }

    auto tensor = context.getTensorToEvaluate();
    AXON_DCHECK(tensor != nullptr);

    auto rank = tensor->rank();

    auto return_descriptor =
        reinterpret_cast<abi::MemRefDescriptor*>(descriptors_.back());

    auto storage_ptr =
        std::make_unique<Storage>(abi::MemRefDescriptor::createStorage(
            return_descriptor, DataType::Float32, rank));

    tensor->setStorage(std::move(storage_ptr));
  }

 private:
  auto compileModuleAndPrepareEngine(GraphContext& context) -> void {
    mlir::OpBuilder builder(mlir_context_);
    mlir::PassManager pm(mlir_context_);

    pm.enableVerifier();

    module_ = codegenGraph(context.graph(), builder);
    axon::buildLlvmLoweringPipeline(pm, LoweringLevel::LLVM);

    auto lowering_result = pm.run(module_);
    if (lowering_result.failed()) {
      throw std::runtime_error("Failed to compile graph.");
    }

    auto opt_pipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);

    mlir::ExecutionEngineOptions engine_options;
    engine_options.transformer = opt_pipeline;

    auto maybe_engine = mlir::ExecutionEngine::create(module_, engine_options);
    if (!maybe_engine) {
      throw std::runtime_error("Failed to create JIT engine");
    }

    std::println("Created engine.");
    engine_ = std::move(maybe_engine.get());
  }

  auto configureDescriptors(GraphContext& context) -> void {
    // If this is the first time we're configuring the descriptors then we need
    // to create them first.
    if (descriptors_.empty()) {
      AXON_DCHECK(context.parameters().size() > 0);

      for (auto tensor : context.parameters()) {
        auto ptr =
            reinterpret_cast<void*>(abi::TensorDescriptor::create(*tensor));
        descriptors_.emplace_back(ptr);
      }

      auto tensor = context.getTensorToEvaluate();
      AXON_DCHECK(tensor != nullptr);

      auto returned_descriptor = abi::MemRefDescriptor::create(
          nullptr, nullptr, 0, tensor->shape(),
          computeStrides(tensor->shape(), Layout::RowMajor));

      descriptors_.emplace_back(reinterpret_cast<void*>(returned_descriptor));
      return;
    }

    // AXON_DCHECK(descriptors_.size() == context.graph().parameters().size() +
    // 1);
    for (auto [tensor, ptr] :
         std::views::zip(context.insts().keys(), descriptors_)) {
      AXON_DCHECK(tensor != nullptr);
      AXON_DCHECK(tensor->isEvaluated());

      auto descriptor = reinterpret_cast<abi::TensorDescriptor*>(ptr);
      abi::TensorDescriptor::setStorage(descriptor, *tensor);
    }
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
  llvm::SmallVector<void*> descriptors_;
  mlir::MLIRContext* mlir_context_;
  mlir::ModuleOp module_;
};

export class GlobalContext {
 public:
  GlobalContext(const GlobalContext&) = delete;
  auto operator=(const GlobalContext&) -> GlobalContext& = delete;

  static auto get() -> GlobalContext& {
    // FIXME:
    // Remove this intentional leak memory since llvm has complex internal
    // destructors, that will result to a std::terminate if not done properly.
    static auto context = new GlobalContext();
    return *context;
  }

  auto execute(GraphContext& context) -> void {
    auto& graph = context.graph();
    if (graph_registry_.contains(graph)) {
      return graph_registry_[graph]->execute(context);
    }

    auto compiled_function = std::make_unique<CompiledFunction>(&mlir_context_);
    compiled_function->execute(context);
    graph_registry_[graph] = std::move(compiled_function);
  }

 private:
  GlobalContext() : mlir_context_(createDialectRegistry()) {
    mlir_context_.loadAllAvailableDialects();
  }

  mlir::MLIRContext mlir_context_;
  llvm::DenseMap<Graph, std::unique_ptr<CompiledFunction>> graph_registry_;
};

}  // namespace axon
