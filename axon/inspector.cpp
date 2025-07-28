

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

import axon.core;
import axon.mlir;

auto main() -> int {
  axon::Module module;

  axon::InstId x = module.declare_input_tensor({2, 3}, true);
  axon::InstId y = module.declare_input_tensor({2, 3}, true);
  axon::InstId l = module.emit_inst(axon::insts::Mul(x, y));
  module.create_return(l);

  mlir::OpPrintingFlags flags;
  mlir::MLIRContext ctx;
  flags.printGenericOpForm(false);

  if (auto mlir_module = axon::codegen(ctx, module)) {
    mlir_module->print(llvm::outs(), flags);
  }
}
