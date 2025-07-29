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

namespace axon {

export class Context {
 public:
  Context(Module& module, mlir::MLIRContext& ctx)
      : builder_(&ctx), module_(module) {
    mlir_module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
  }

  auto builder() -> mlir::OpBuilder& { return builder_; }
  auto builder() const -> const mlir::OpBuilder& { return builder_; }

  auto module() -> Module& { return module_; }
  auto module() const -> const Module& { return module_; }

  auto mlir_module() -> mlir::ModuleOp& { return mlir_module_; }
  auto mlir_module() const -> const mlir::ModuleOp& { return mlir_module_; }

  auto tensors(bool is_forward) -> llvm::DenseMap<InstId, mlir::Value>& {
    return is_forward ? forward_tensors_ : backward_tensors_;
  }

  auto inputs(bool is_forward) -> llvm::DenseMap<InputId, mlir::Value>& {
    return is_forward ? forward_inputs_ : backward_inputs_;
  }

  auto cached_values(bool is_forward)
      -> llvm::DenseMap<CachedValueId, mlir::Value>& {
    return is_forward ? forward_cached_values_ : backward_cached_values_;
  }

  template <Index T>
  consteval auto get_argument_index() -> int {
    if constexpr (std::is_same_v<T, InputId>) {
      return 0;
    } else if constexpr (std::is_same_v<T, CachedValueId>) {
      return 1;
    } else {
      static_assert(false, "Unreachable");
    }
  }

  template <Index T>
  auto get_argument_map(bool is_forward) -> llvm::DenseMap<T, mlir::Value>& {
    if constexpr (std::is_same_v<T, InputId>) {
      return is_forward ? forward_inputs_ : backward_inputs_;
    } else if constexpr (std::is_same_v<T, CachedValueId>) {
      return is_forward ? forward_cached_values_ : backward_cached_values_;
    } else {
      static_assert(false, "Unreachable");
    }
  }

 private:
  mlir::OpBuilder builder_;
  mlir::ModuleOp mlir_module_;
  Module& module_;

  llvm::DenseMap<InstId, mlir::Value> forward_tensors_;
  llvm::DenseMap<InstId, mlir::Value> backward_tensors_;

  llvm::DenseMap<InputId, mlir::Value> forward_inputs_;
  llvm::DenseMap<InputId, mlir::Value> backward_inputs_;

  llvm::DenseMap<CachedValueId, mlir::Value> forward_cached_values_;
  llvm::DenseMap<CachedValueId, mlir::Value> backward_cached_values_;
};

}  // namespace axon
