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
import axon.base;

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

    // if (mlir::failed(mlir::verify(mlir_module))) {
    //   mlir_module.emitError("module verification error");
    //   return nullptr;
    // }

    return mlir_module;
  }

 private:
  // auto get_parameter_list() -> TensorRefListType {
  //   llvm::SmallVector<TensorRefType> params;
  //   auto* context = builder_.getContext();
  //   for (auto [param_id, param] : module_.parameters().iter_values()) {
  //     auto param_type = TensorRefType::get(context, builder_.getF32Type(),
  //                                          param.shape, param.requires_grad);
  //     params.push_back(param_type);
  //   }
  //   return TensorRefListType::get(context, params);
  // }

  auto codegen_forward() -> void {
    llvm::SmallVector<mlir::Type> input_types;

    // input_types.push_back(get_parameter_list());

    // The return type will be inferred later.
    auto func_type = builder_.getFunctionType(input_types, {});
    auto func = builder_.create<mlir::func::FuncOp>(builder_.getUnknownLoc(),
                                                    "forward", func_type);
    auto* entry_block = func.addEntryBlock();
    builder_.setInsertionPointToStart(entry_block);

    std::flat_map<InstId, mlir::Value> values;
    // for (auto [inst_id, inst] : module_.forward_insts().iter_values()) {
    //   codegen_instruction(inst_id, inst, values, entry_block);
    // }

    builder_.create<mlir::func::ReturnOp>(builder_.getUnknownLoc());
  }

  // auto codegen_instruction(instid inst_id, const inst& inst,
  //                          std::flat_map<instid, mlir::value>& values,
  //                          mlir::block* block) -> void {
  //   auto loc = builder_.getUnknownLoc();
  //   // if (auto get_param = inst.try_get_as<insts::GetParameter>()) {
  //   //   auto param_list =
  //   // llvm::dyn_cast<TensorRefListType>(block->getArgument(0).getType());
  //   //
  //   //   auto index = get_param->param_id.value();
  //   //   auto result_type = param_list.getValues()[index];
  //   //
  //   //   values[inst_id] = builder_.create<ListAccessOp>(
  //   //       loc, result_type, block->getArgument(0), index);
  //   //   return;
  //   // }
  // }

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
