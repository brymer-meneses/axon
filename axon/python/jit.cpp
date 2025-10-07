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
import :storage;
import :abi;

namespace axon {

class CompiledFunction {
 public:
  CompiledFunction(mlir::MLIRContext* context, const Graph& graph,
                   llvm::ArrayRef<Tensor*> returns)
      : context_(context) {
    compileModuleAndPrepareEngine(graph, returns);

    auto max_size = 2 * (graph.parameters().size()) + returns.size();

    argv_.reserve(max_size);
    arg_slots_.reserve(max_size);
  }

  auto execute(llvm::ArrayRef<std::shared_ptr<Tensor>> parameters,
               llvm::ArrayRef<Tensor*> returns) -> void {
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

    llvm::SmallVector<std::unique_ptr<Storage>> storages;
    for (auto& returned : returns) {
      auto cpu =
          CpuStorage::createUninit(returned->shape(), returned->data_type());
      auto storage = std::make_unique<Storage>(Storage(std::move(cpu)));

      arg_slots_.push_back(storage->data_ptr());
      argv_.push_back(reinterpret_cast<void*>(&arg_slots_.back()));

      storages.push_back(std::move(storage));
    }

    auto invocation_result = engine_->invokePacked("graph", argv_);
    if (invocation_result) {
      throw std::runtime_error("JIT invocation failed");
    }

    for (auto i = 0; i < storages.size(); i += 1) {
      auto& returned = returns[i];
      auto& storage = storages[i];

      returned->setStorage(std::move(storage));
    }
  }

 private:
  auto compileModuleAndPrepareEngine(const Graph& graph,
                                     llvm::ArrayRef<Tensor*> returns) -> void {
    mlir::OpBuilder builder(context_);
    mlir::PassManager pm(context_);

    pm.enableVerifier();

    auto return_ids = getInstIds(returns);

    mlir::ModuleOp module = codegenGraph(graph, return_ids, builder);
    axon::buildLlvmLoweringPipeline(pm, LoweringLevel::LLVM);

    auto lowering_result = pm.run(module);
    if (lowering_result.failed()) {
      module.print(llvm::outs());
      throw std::runtime_error("Failed to compile graph.");
    }

    auto opt_pipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);

    mlir::ExecutionEngineOptions engine_options;
    engine_options.transformer = opt_pipeline;

    auto maybe_engine = mlir::ExecutionEngine::create(module, engine_options);
    if (!maybe_engine) {
      module.print(llvm::outs());
      throw std::runtime_error("Failed to create JIT engine");
    }

    engine_ = std::move(maybe_engine.get());
  }

  static auto getInstIds(llvm::ArrayRef<Tensor*> tensors)
      -> llvm::SmallVector<InstId> {
    llvm::SmallVector<InstId> inst_ids;
    for (auto tensor : tensors) {
      AXON_DCHECK(tensor);

      auto inst_id = tensor->getInstId();
      AXON_DCHECK(inst_id.isValid());

      inst_ids.push_back(inst_id);
    }

    return inst_ids;
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;

  mlir::MLIRContext* context_;

  llvm::SmallVector<void*> argv_;
  llvm::SmallVector<std::byte*> arg_slots_;
};

}  // namespace axon
