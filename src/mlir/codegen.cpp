module;

#include <flat_map>
#include <print>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

export module axon.mlir;

import axon.core;
import axon.base.value_store;

namespace axon {

class Context {
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
  auto codegen_forward() -> mlir::func::FuncOp {
    llvm::SmallVector<mlir::Type> input_types;
    std::flat_map<InstId, mlir::Value> values;

    for (auto [param_id, param] : module_.parameters().iter_values()) {
      auto tensor_type =
          mlir::RankedTensorType::get(param.shape, builder_.getF32Type());
      input_types.push_back(tensor_type);
    }

    // The return type will be inferred later.
    auto func_type = builder_.getFunctionType(input_types, {});
    auto func = builder_.create<mlir::func::FuncOp>(builder_.getUnknownLoc(),
                                                    "forward", func_type);

    auto* entry_block = func.addEntryBlock();
    builder_.setInsertionPointToStart(entry_block);

    for (auto [inst_id, inst] : module_.forward_insts().iter_values()) {
      codegen_instruction(inst_id, inst, values, entry_block);
    }

    // Add a return statement (functions need to return something)
    builder_.create<mlir::func::ReturnOp>(builder_.getUnknownLoc());
    return func;
  }

  auto codegen_instruction(InstId inst_id, const Inst& inst,
                           std::flat_map<InstId, mlir::Value>& values,
                           mlir::Block* block) -> void {
    if (auto get_param = inst.try_get_as<insts::GetParameter>()) {
      values[inst_id] = block->getArgument(get_param->param_id.value());
      return;
    }

    if (auto add = inst.try_get_as<insts::Add>()) {
      auto lhs = values[add->lhs_id];
      auto rhs = values[add->rhs_id];
      auto result = builder_.create<mlir::arith::AddFOp>(
          builder_.getUnknownLoc(), lhs, rhs);
      values[inst_id] = result;
      return;
    }

    if (auto mul = inst.try_get_as<insts::Mul>()) {
      auto lhs = values[mul->lhs_id];
      auto rhs = values[mul->rhs_id];
      auto result = builder_.create<mlir::arith::MulFOp>(
          builder_.getUnknownLoc(), lhs, rhs);
      values[inst_id] = result;
      return;
    }

    if (auto matmul = inst.try_get_as<insts::MatMul>()) {
      auto lhs = values[matmul->lhs_id];
      auto rhs = values[matmul->rhs_id];
      auto result = builder_.create<mlir::arith::MulFOp>(
          builder_.getUnknownLoc(), lhs, rhs);
      values[inst_id] = result;
      return;
    }
  }

 private:
  mlir::OpBuilder builder_;
  const Module& module_;
};

export auto codegen(mlir::MLIRContext& ctx, const Module& module)
    -> mlir::ModuleOp {
  Context context{ctx, module};
  return context.codegen();
}

};  // namespace axon
