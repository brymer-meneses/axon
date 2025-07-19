

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

import axon.core;
import axon.mlir;

auto main() -> int {
  axon::Module module;

  auto x = module.create_tensor({{1, 2, 3}, {4, 5, 6}}, true);
  auto y = module.create_tensor({{1, 2, 3}, {1, 2, 3}}, true);
  auto l = module.emit_inst(axon::insts::Mul(x, y));
  module.create_return(l);

  mlir::OpPrintingFlags flags;
  mlir::MLIRContext ctx;
  flags.printGenericOpForm(false);

  if (auto mlir_module = axon::codegen(ctx, module)) {
    mlir_module->print(llvm::outs(), flags);
  }
}
