module;

#include <flat_map>
#include <print>

#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
export module axon.mlir;

import axon.core;

import axon.base.value_store;

namespace axon {

class Context {
  static constexpr int32_t ParamsValuesIndex = 0;
  static constexpr int32_t CachedValuesIndex = 1;

 public:
  Context(mlir::MLIRContext& context, const Module& module)
      : builder_(&context), module_(module) {}

  auto codegen() -> mlir::ModuleOp {
    auto mlir_module = mlir::ModuleOp::create(builder_.getUnknownLoc());
    builder_.setInsertionPointToEnd(mlir_module.getBody());
    codegen_forward();

    if (failed(mlir::verify(mlir_module))) {
      mlir_module.emitError("module verification error");
      return nullptr;
    }

    return mlir_module;
  }

 private:
  auto codegen_forward() -> void {
    llvm::SmallVector<mlir::Type> input_types;
    std::flat_map<InstId, mlir::Value> values;

    for (auto [param_id, param] : module_.parameters().iter_values()) {
      auto tensor_type = ParameterType::get(param.shape, builder_.getF32Type());

      input_types.push_back(tensor_type);
    }

    // The return type will be inferred later.
    auto func_type = builder_.getFunctionType(input_types, {});
    auto func = builder_.create<mlir::func::FuncOp>(builder_.getUnknownLoc(),
                                                    "forward", func_type);

    auto* entry_block = func.addEntryBlock();
    builder_.setInsertionPointToStart(entry_block);

    builder_.create<mlir::func::ReturnOp>(builder_.getUnknownLoc());
  }

 private:
  mlir::OpBuilder builder_;
  [[maybe_unused]] const Module& module_;
};

export auto codegen(mlir::MLIRContext& ctx, const Module& module)
    -> mlir::ModuleOp {
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::tensor::TensorDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();
  ctx.loadDialect<AxonDialect>();

  Context context{ctx, module};
  return context.codegen();
}

};  // namespace axon
