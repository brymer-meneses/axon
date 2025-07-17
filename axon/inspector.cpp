
#include <iostream>
#include <print>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

import axon.core;
import axon.mlir;

auto main() -> int {
  axon::Module module;

  auto x = module.create_tensor({10, 10}, true);
  auto y = module.create_tensor({10, 10}, true);
  auto z = module.emit_inst(axon::insts::Mul(x, y));
  auto l = module.emit_inst(axon::insts::Add(z, y));

  module.create_return(l);

  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm(false);

  if (auto mlir_module = axon::codegen(module)) {
    mlir_module->print(llvm::outs(), flags);
  }
}
