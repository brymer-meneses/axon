module;

#include <memory>
#include <stdexcept>
#include <vector>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

export module axon.python:compilation_unit;

import axon.core;
import axon.mlir;

import :tensor;
import :abi;

namespace axon {

static auto processParams(Graph& graph,
                          std::span<std::shared_ptr<Tensor>> params)
    -> llvm::SmallVector<void*> {
  llvm::SmallVector<void*> args;

  for (auto tensor : params) {
    auto ptr = reinterpret_cast<void*>(abi::TensorDescriptor::create(*tensor));
    args.emplace_back(ptr);
  }

  auto returned_id = graph.getReturnedId();
  if (!returned_id) {
    return args;
  }

  auto returned_descriptor =
      abi::MemRefDescriptor::createEmpty(graph.getShape(returned_id).size());
  args.emplace_back(reinterpret_cast<void*>(returned_descriptor));
  return args;
}

static auto destroyParams(llvm::ArrayRef<void*> params) -> void {
  for (auto param : params) {
    auto tensor_descriptor = reinterpret_cast<abi::TensorDescriptor*>(param);
    abi::TensorDescriptor::destroy(tensor_descriptor);
  }
}

export class CompilationUnit {
 public:
  CompilationUnit(Graph& graph)
      : context_(createDialectRegistry()), graph_(graph) {
    context_.loadAllAvailableDialects();
  }

  auto compile(LoweringLevel level) -> mlir::LogicalResult {
    mlir::PassManager manager(&context_);
    mlir::OpBuilder builder(&context_);

    module_op_ = mlir::ModuleOp::create(builder.getUnknownLoc());
    codegenGraph(graph_, builder, module_op_);

    if (mlir::verify(module_op_).failed()) {
      module_op_.walk([&](mlir::Operation* op) {
        if (mlir::verify(op).failed()) {
          llvm::errs() << "Verifier failed on op: " << op->getName() << "\n";
          op->dump();
        }
      });
      return mlir::failure();
    }

    manager.enableVerifier();

    axon::buildLlvmLoweringPipeline(manager, level);

    auto result = manager.run(module_op_);
    return result;
  }

  auto execute(std::vector<std::shared_ptr<Tensor>> params)
      -> std::optional<Tensor> {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto opt_pipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);

    mlir::ExecutionEngineOptions engine_options;
    engine_options.transformer = opt_pipeline;
    auto maybe_engine =
        mlir::ExecutionEngine::create(module_op_, engine_options);
    if (!maybe_engine) {
      throw std::runtime_error("Failed to create JIT engine");
    }

    auto& engine = maybe_engine.get();
    auto args = processParams(graph_, params);

    auto invocation_result = engine->invokePacked("graph", args);
    if (invocation_result) {
      throw std::runtime_error("JIT invocation failed\n");
    }

    auto returned_id = graph_.getReturnedId();
    if (!returned_id) {
      // destroyParams(args);
      return std::nullopt;
    }

    auto rank = graph_.getShape(returned_id).size();
    auto return_descriptor =
        reinterpret_cast<abi::MemRefDescriptor*>(args.back());
    auto storage = abi::MemRefDescriptor::createStorage(
        return_descriptor, DataType::Float32, rank);
    destroyParams(args);

    return {Tensor(std::move(storage), /*requires_grad=*/false)};
  }

  auto dump_ir() const -> std::string {
    std::string repr;
    llvm::raw_string_ostream string_stream{repr};
    module_op_->print(string_stream);
    return repr;
  }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleOp module_op_;
  Graph& graph_;
};

}  // namespace axon
