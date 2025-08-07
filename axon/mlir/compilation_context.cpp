module;

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

export module axon.mlir:compilation_context;

import axon.core;
import axon.base;

export namespace axon {

struct CompilationContext {
  Graph& graph;
  mlir::OpBuilder& builder;
  mlir::func::FuncOp& func_op;

  llvm::DenseMap<InstId, mlir::Value> tensor_refs{};
  llvm::DenseMap<InstId, mlir::Value> values{};
};

}  // namespace axon
