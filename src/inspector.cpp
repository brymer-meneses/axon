
#include <iostream>
#include <print>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

import axon.core;
import axon.mlir;

auto main() -> int {
  axon::Module module;

  auto x = module.declare_parameter("x", {10, 10}, true);
  auto y = module.declare_parameter("y", {10, 10}, true);
  auto z = module.create_inst(axon::insts::Mul(x, y));

  module.build_backward(z);

  mlir::MLIRContext ctx;
  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm(false);

  if (auto mlir_module = axon::codegen(ctx, module)) {
    mlir_module->print(llvm::outs(), flags);
  }
}
