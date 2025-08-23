module;

#include <optional>

#include "mlir/IR/BuiltinOps.h"

export module axon.python:compilation_unit;

import axon.core;
import axon.mlir;

import :storage;
import :tensor;

export namespace axon {

class CompilationUnit {
 public:
  CompilationUnit() : builder_(&context_) {
    module_op_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
  }

  auto compile() -> void {}

  auto execute() -> void {}

 private:
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_op_;
};

}  // namespace axon
