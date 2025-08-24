module;

#include <optional>
#include <print>

#include "mlir/IR/BuiltinOps.h"

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

  auto module_op() -> auto& { return module_op_; }
  auto module_op() const -> const auto& { return module_op_; }

 private:
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_op_;
};

}  // namespace axon
