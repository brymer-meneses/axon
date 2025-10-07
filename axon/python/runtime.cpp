module;

#include <atomic>
#include <memory>
#include <print>
#include <unordered_map>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/MLIRContext.h"

export module axon.python:runtime;

import axon.core;

import :tensor;
import :abi;
import :jit;

namespace axon {

export class Runtime {
 public:
  Runtime(const Runtime&) = delete;
  auto operator=(const Runtime&) -> Runtime& = delete;

  static auto get() -> Runtime& {
    // FIXME:
    // Remove this intentional leak memory since llvm has complex internal
    // destructors, that will result to a std::terminate if not done properly.
    static auto context = new Runtime();
    return *context;
  }

  auto execute(const Graph& graph,
               llvm::ArrayRef<std::shared_ptr<Tensor>> parameters,
               llvm::ArrayRef<Tensor*> returns) {
    auto hash = graph.hash();
    if (graph_registry_.contains(graph)) {
      return graph_registry_[graph]->execute(parameters, returns);
    }

    auto compiled_function =
        std::make_unique<CompiledFunction>(&mlir_context_, graph, returns);
    compiled_function->execute(parameters, returns);
    graph_registry_[graph] = std::move(compiled_function);
  }

  auto getTotalNumberOfCompiledFunctions() const -> u64 {
    return graph_registry_.size();
  }

  auto shouldEmitGrad() const -> bool { return emit_grad_; }
  auto setEmitGrad(bool value) -> void { emit_grad_ = value; }

 private:
  Runtime() : mlir_context_(createDialectRegistry()) {
    mlir_context_.loadAllAvailableDialects();
  }

  mlir::MLIRContext mlir_context_;
  std::unordered_map<Graph, std::unique_ptr<CompiledFunction>> graph_registry_;

  inline static thread_local bool emit_grad_ = true;
};

}  // namespace axon
