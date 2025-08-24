module;

#include <print>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

export module axon.python:compilation_unit;

import axon.core;
import axon.mlir;

import :tensor;

export namespace axon {

class CompilationUnit {
 public:
  CompilationUnit() : builder_(&context_) {
    module_op_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
  }

  auto compile(Graph& graph) -> void { module_op_ = codegen(graph, context_); }

  auto runToStandardPass() -> void {
    mlir::PassManager manager(&context_);
    manager.enableVerifier();

    manager.addPass(axon::createStandardLoweringPass());

    auto result = manager.run(module_op_);
    if (result.failed()) {
      std::println("Pass failed");
    }
  }

  auto module_op() -> auto& { return module_op_; }
  auto module_op() const -> const auto& { return module_op_; }

 private:
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_op_;
};

}  // namespace axon
