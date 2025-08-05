
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

import axon.core;
import axon.mlir;
import std;

auto main() -> int {
  auto ctx = std::make_shared<axon::Context>();
  axon::Module mod{ctx};

  auto x = mod.declareInput({2, 3}, true);
  auto y = mod.declareInput({2, 3}, true);
  auto l = mod.emit(axon::insts::Mul(x, y));
  mod.createReturn(l);

  axon::finalize(mod);

  mlir::OpPrintingFlags flags;
  mlir::MLIRContext mlir_context;

  flags.printGenericOpForm(false);

  if (auto module_op = axon::codegen(mod, mlir_context)) {
    module_op->print(llvm::outs(), flags);
  }
}
