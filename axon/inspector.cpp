
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

import axon.core;
import axon.mlir;
import std;

auto main() -> int {
  axon::Graph graph;

  auto x = graph.declareParam({2, 3}, true);
  auto y = graph.declareParam({2, 3}, true);
  auto l = graph.createOp(axon::insts::Mul(x, y));

  axon::backward(graph, l);

  mlir::OpPrintingFlags flags;
  mlir::MLIRContext mlir_context;

  flags.printGenericOpForm(false);

  if (auto module_op = axon::codegen(graph, mlir_context)) {
    module_op->print(llvm::outs(), flags);
  }
}
