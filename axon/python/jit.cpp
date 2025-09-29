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

  auto execute(llvm::ArrayRef<Tensor*> parameters, Tensor* returned = nullptr)
      -> void {
    // context contains a graph that is isomorphic to the one that was used to
    // construct this CompiledFunction.
    configureDescriptors(parameters, returned);

    auto invocation_result = engine_->invokePacked("graph", descriptors_);
    if (invocation_result) {
      throw std::runtime_error("JIT invocation failed");
    }

    if (returned) {
      auto rank = returned->rank();
      auto return_descriptor =
          reinterpret_cast<abi::MemRefDescriptor*>(descriptors_.back());

      auto storage =
          std::make_unique<Storage>(abi::MemRefDescriptor::createStorage(
              return_descriptor, returned->data_type(), rank));

      returned->setStorage(std::move(storage));
    }
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
      module_.print(llvm::outs());
      throw std::runtime_error("Failed to create JIT engine");
    }

    engine_ = std::move(maybe_engine.get());
  }

  auto initializeDescriptors(llvm::ArrayRef<Tensor*> parameters,
                             Tensor* returned_value) -> void {
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

    if (returned_value) {
      auto returned_descriptor = abi::MemRefDescriptor::create(
          nullptr, nullptr, 0, returned_value->shape(),
          computeStrides(returned_value->shape(), Layout::RowMajor));

      descriptors_.emplace_back(reinterpret_cast<void*>(returned_descriptor));
      has_returned_ = true;
    }
  }

  auto configureDescriptors(llvm::ArrayRef<Tensor*> parameters,
                            Tensor* returned_value) -> void {
    if (descriptors_.empty()) {
      initializeDescriptors(parameters, returned_value);
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

    if (returned_value) {
      AXON_DCHECK(
          has_returned_,
          "This function has been configured to have no returned value but "
          "passed returned != nullptr");

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
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
  llvm::SmallVector<void*> descriptors_;
  mlir::MLIRContext* context_;
  mlir::ModuleOp module_;

  bool has_returned_ = false;
};

}  // namespace axon
