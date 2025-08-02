module;

#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

export module axon.mlir:compilation_context;

import axon.core;
import axon.base;

export namespace axon {

class CompilationContext {
 public:
  CompilationContext(Module& module, mlir::MLIRContext& ctx)
      : builder_(&ctx), module_(module) {
    module_op_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
  }

  auto builder() -> auto& { return builder_; }
  auto module() -> auto& { return module_; }
  auto module_op() -> auto& { return module_op_; }

 private:
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_op_;
  Module& module_;
};

}  // namespace axon
