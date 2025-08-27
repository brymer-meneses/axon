#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
module;

#include <memory>
#include <print>
#include <vector>

#include "axon/base/dcheck.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

export module axon.python:compilation_unit;

import axon.core;
import axon.mlir;

import :tensor;

namespace axon {

export class CompilationUnit {
 public:
  CompilationUnit() : builder_(&context_) {
    module_op_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
  }

  auto compile(Graph& graph, LoweringOps ops) -> mlir::LogicalResult {
    mlir::PassManager manager(&context_);
    module_op_ = codegen(graph, context_);
    manager.enableVerifier();

    axon::buildLlvmLoweringPipeline(manager, ops);

    auto result = manager.run(module_op_);
    return result;
  }

  auto execute(std::vector<std::shared_ptr<Tensor>> params) -> void {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::registerBuiltinDialectTranslation(context_);
    mlir::registerLLVMDialectTranslation(context_);

    auto opt_pipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);

    mlir::ExecutionEngineOptions engine_options;
    engine_options.transformer = opt_pipeline;
    auto maybe_engine =
        mlir::ExecutionEngine::create(module_op_, engine_options);
    AXON_DCHECK(maybe_engine, "failed to construct an execution engine");
    auto& engine = maybe_engine.get();

    std::println("begin");
    auto invocation_result = engine->invokePacked("graph");
    if (invocation_result) {
      llvm::errs() << "JIT invocation failed\n";
    }
    std::println("end");
  }

  auto dump_ir() const -> std::string {
    std::string repr;
    llvm::raw_string_ostream string_stream{repr};
    module_op_->print(string_stream);
    return repr;
  }

 private:
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_op_;
};

}  // namespace axon
