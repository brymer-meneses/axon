module;

#include <memory>
#include <print>
#include <ranges>
#include <unordered_map>

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
  CompiledFunction(mlir::MLIRContext* context, Graph& graph)
      : context_(context) {
    compileModuleAndPrepareEngine(graph);
  }

  ~CompiledFunction() {
    if (descriptors_.empty()) {
      return;
    }

    auto last_index = descriptors_.size() - 1;
    for (size_t i = 0; i < last_index; ++i) {
      auto tensor_descriptor =
          reinterpret_cast<abi::TensorDescriptor*>(descriptors_[i]);
      abi::TensorDescriptor::destroy(tensor_descriptor);
    }

    auto returned_descriptor =
        reinterpret_cast<abi::MemRefDescriptor*>(descriptors_.back());
    abi::MemRefDescriptor::destroy(returned_descriptor);
  }

  auto execute(llvm::ArrayRef<Tensor*> parameters, Tensor* returned)
      -> std::unique_ptr<Storage> {
    // context contains a graph that is isomorphic to the one that was used to
    // construct this CompiledFunction.
    configureDescriptors(parameters, returned);

    auto invocation_result = engine_->invokePacked("graph", descriptors_);
    if (invocation_result) {
      throw std::runtime_error("JIT invocation failed");
    }

    auto rank = returned->rank();
    auto return_descriptor =
        reinterpret_cast<abi::MemRefDescriptor*>(descriptors_.back());

    return std::make_unique<Storage>(abi::MemRefDescriptor::createStorage(
        return_descriptor, returned->data_type(), rank));
  }

 private:
  auto compileModuleAndPrepareEngine(Graph& graph) -> void {
    mlir::OpBuilder builder(context_);
    mlir::PassManager pm(context_);

    pm.enableVerifier();

    module_ = codegenGraph(graph, builder);
    axon::buildLlvmLoweringPipeline(pm, LoweringLevel::LLVM);

    auto lowering_result = pm.run(module_);
    if (lowering_result.failed()) {
      module_.print(llvm::outs());
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

    engine_ = std::move(maybe_engine.get());
  }

  auto configureDescriptors(llvm::ArrayRef<Tensor*> parameters,
                            Tensor* returned_value) -> void {
    // If this is the first time we're configuring the descriptors then we need
    // to create them first.
    if (descriptors_.empty()) {
      AXON_DCHECK(parameters.size() > 0);

      for (auto tensor : parameters) {
        AXON_DCHECK(tensor != nullptr);

        if (tensor->requiresGrad() && tensor->grad() == nullptr) {
          tensor->zeroGrad();
        }

        auto ptr =
            reinterpret_cast<void*>(abi::TensorDescriptor::create(*tensor));
        descriptors_.emplace_back(ptr);
      }

      AXON_DCHECK(returned_value != nullptr);

      auto returned_descriptor = abi::MemRefDescriptor::create(
          nullptr, nullptr, 0, returned_value->shape(),
          computeStrides(returned_value->shape(), Layout::RowMajor));

      descriptors_.emplace_back(reinterpret_cast<void*>(returned_descriptor));
      return;
    }

    AXON_DCHECK(returned_value != nullptr);
    AXON_DCHECK(descriptors_.size() == parameters.size() + 1);

    for (auto [tensor, ptr] : std::views::zip(parameters, descriptors_)) {
      AXON_DCHECK(tensor != nullptr);
      AXON_DCHECK(tensor->isEvaluated());

      if (tensor->requiresGrad() && tensor->grad() == nullptr) {
        tensor->zeroGrad();
      }

      auto descriptor = reinterpret_cast<abi::TensorDescriptor*>(ptr);
      abi::TensorDescriptor::setStorage(descriptor, *tensor);
    }

    auto returned_descriptor =
        reinterpret_cast<abi::MemRefDescriptor*>(descriptors_.back());

    auto shape = returned_value->shape();
    auto strides = computeStrides(shape, Layout::RowMajor);

    abi::MemRefDescriptor::createInPlace(
        reinterpret_cast<std::byte*>(returned_descriptor),
        /*allocated_ptr=*/nullptr,
        /*aligned_ptr=*/nullptr,
        /*offset=*/0, shape, strides);
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
  llvm::SmallVector<void*> descriptors_;
  mlir::MLIRContext* context_;
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

  auto execute(llvm::ArrayRef<Tensor*> parameters, Tensor* returned,
               Graph& graph) -> std::unique_ptr<Storage> {
    auto hash = graph.hash();

    if (graph_registry_.contains(graph)) {
      std::println("reusing hash {}", hash);
      return graph_registry_[graph]->execute(parameters, returned);
    }

    std::println("creating function with hash {}", hash);
    auto compiled_function =
        std::make_unique<CompiledFunction>(&mlir_context_, graph);
    auto storage = compiled_function->execute(parameters, returned);
    graph_registry_[graph] = std::move(compiled_function);
    return std::move(storage);
  }

 private:
  GlobalContext() : mlir_context_(createDialectRegistry()) {
    mlir_context_.loadAllAvailableDialects();
  }

  mlir::MLIRContext mlir_context_;
  std::unordered_map<Graph, std::unique_ptr<CompiledFunction>> graph_registry_;
};

}  // namespace axon
