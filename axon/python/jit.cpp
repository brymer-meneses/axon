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

  auto execute(llvm::ArrayRef<std::shared_ptr<Tensor>> parameters,
               Tensor* returned = nullptr) -> void {
    argv_.clear();
    arg_slots_.clear();

    for (auto& tensor_ptr : parameters) {
      auto* tensor = tensor_ptr.get();
      auto* data_ptr = tensor->storage()->data_ptr();

      arg_slots_.push_back(data_ptr);
      argv_.push_back(reinterpret_cast<void*>(&arg_slots_.back()));

      if (tensor->requiresGrad()) {
        auto* gptr = tensor->grad()->storage()->data_ptr();
        arg_slots_.push_back(gptr);
        argv_.push_back(reinterpret_cast<void*>(&arg_slots_.back()));
      }
    }

    std::unique_ptr<Storage> storage = nullptr;
    if (returned) {
      storage = std::make_unique<Storage>(
          Storage::createUninit(returned->shape(), returned->data_type()));

      arg_slots_.push_back(storage->data_ptr());
      argv_.push_back(reinterpret_cast<void*>(&arg_slots_.back()));
    }

    auto invocation_result = engine_->invokePacked("graph", argv_);
    if (invocation_result) {
      throw std::runtime_error("JIT invocation failed");
    }

    if (returned) {
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

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;

  mlir::MLIRContext* context_;
  mlir::ModuleOp module_;

  llvm::SmallVector<void*> argv_;
  llvm::SmallVector<std::byte*> arg_slots_;
};

}  // namespace axon
