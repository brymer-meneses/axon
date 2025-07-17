module;

#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

export module axon.mlir:context;

import axon.core;

namespace axon {

export class Context {
 public:
  Context(Module& module) : builder_(&context_), module_(module) {
    context_.loadDialect<mlir::func::FuncDialect>();
    context_.loadDialect<mlir::tensor::TensorDialect>();
    context_.loadDialect<mlir::arith::ArithDialect>();
    context_.loadDialect<AxonDialect>();

    mlir_module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
  }

  auto builder() -> mlir::OpBuilder& { return builder_; }
  auto builder() const -> const mlir::OpBuilder& { return builder_; }

  auto module() -> Module& { return module_; }
  auto module() const -> const Module& { return module_; }

  auto mlir_module() -> mlir::ModuleOp& { return mlir_module_; }
  auto mlir_module() const -> const mlir::ModuleOp& { return mlir_module_; }

  auto forward_values() -> llvm::DenseMap<InstId, mlir::Value>& {
    return forward_values_;
  }
  auto forward_values() const -> const llvm::DenseMap<InstId, mlir::Value>& {
    return forward_values_;
  }

  auto backward_values() -> llvm::DenseMap<InstId, mlir::Value>& {
    return backward_values_;
  }
  auto backward_values() const -> const llvm::DenseMap<InstId, mlir::Value>& {
    return backward_values_;
  }

 private:
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp mlir_module_;
  Module& module_;

  llvm::DenseMap<InstId, mlir::Value> forward_values_;
  llvm::DenseMap<InstId, mlir::Value> backward_values_;
};

}  // namespace axon
