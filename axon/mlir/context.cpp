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
import axon.base;

export namespace axon::backend {

struct Context {
  Context(core::Module& module, mlir::MLIRContext& ctx)
      : builder(&ctx), module(module) {
    mlir_module = mlir::ModuleOp::create(builder.getUnknownLoc());
  }

  mlir::OpBuilder builder;
  mlir::ModuleOp mlir_module;
  core::Module& module;
};

}  // namespace axon::backend
