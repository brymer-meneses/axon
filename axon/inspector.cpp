
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"

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
  mlir_context.disableMultithreading();

  flags.printGenericOpForm(false);

  auto module_op = axon::codegen(graph, mlir_context);
  if (!module_op) {
    std::println("Failed to compile module");
    return 1;
  }

  module_op->print(llvm::outs(), flags);
  mlir::PassManager manager(&mlir_context);
  manager.enableVerifier();
  manager.enableIRPrinting([](mlir::Pass*, mlir::Operation*) { return true; },
                           [](mlir::Pass*, mlir::Operation*) {
                             return true;
                           },            // shouldPrintAfterPass
                           true,         // printModuleScope
                           true,         // printAfterOnlyOnChange
                           false,        // printAfterOnlyOnFailure
                           llvm::errs()  // output stream
  );
  axon::createLowerToLlvmPipeline(manager);

  auto result = manager.run(module_op);
  if (result.failed()) {
    std::println("Pass pipeline failed");
    return {};
  }

  // module_op->print(llvm::outs(), flags);
}
