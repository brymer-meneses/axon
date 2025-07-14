
#include <iostream>
#include <print>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

import axon.core;
import axon.mlir;

auto main() -> int {
  axon::Module outer_module;
  axon::Module module;

  auto input = outer_module.create_constant_tensor({10, 10}, true);

  auto x = module.create_constant_tensor({10, 10}, true);
  auto y = module.create_constant_tensor({10, 10}, true);
  auto w = module.declare_input_tensor(&outer_module, input);
  auto z = module.track_operation(axon::insts::Mul(x, y));
  auto l = module.track_operation(axon::insts::Add(z, w));

  module.create_return(l);

  mlir::MLIRContext ctx;
  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm(false);

  if (auto mlir_module = axon::codegen(ctx, module)) {
    mlir_module->print(llvm::outs(), flags);
  }
}
