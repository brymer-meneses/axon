

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

import axon.core;
// import axon.mlir;

using namespace axon;

auto main() -> int {
  core::Context ctx;
  core::Module module(ctx);
  auto x = module.forward_function.declare_input({2, 3}, true);
  auto y = module.forward_function.declare_input({2, 3}, true);
  auto l = module.forward_function.emit(core::insts::Mul(x, y));

  module.create_return(l);
  //
  // mlir::OpPrintingFlags flags;
  // mlir::MLIRContext ctx;
  // flags.printGenericOpForm(false);
  //
  // if (auto mlir_module = axon::codegen(ctx, module)) {
  //   mlir_module->print(llvm::outs(), flags);
  // }
}
