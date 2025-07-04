
#include <iostream>
#include <print>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

import axon.core;
import axon.mlir;

auto main() -> int {
  axon::Module module;

  auto x = module.declare_parameter("x", {10, 10}, true);
  auto y = module.declare_parameter("y", {10, 10}, true);
  auto z = module.create_inst(axon::insts::Add(x, y));
  auto w = module.create_inst(axon::insts::Mul(z, z));

  module.build_backward(y);

  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::tensor::TensorDialect>();
  ctx.loadDialect<mlir::arith::ArithDialect>();

  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm(false);

  if (auto mlir_module = axon::codegen(ctx, module)) {
    mlir_module->print(llvm::outs(), flags);
  }
}
