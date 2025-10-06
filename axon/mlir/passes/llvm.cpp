module;

#include <memory>

#include "axon/base/macros.h"
#include "axon/mlir/dialect/dialect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

export module axon.mlir.passes:llvm_lowering;

namespace axon {

struct AxonToLlvmLoweringPass
    : mlir::PassWrapper<AxonToLlvmLoweringPass,
                        mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AxonToLlvmLoweringPass)

  auto getArgument() const -> llvm::StringRef override {
    return "axon-to-llvm-lowering";
  }

  auto getDependentDialects(mlir::DialectRegistry& registry) const
      -> void override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                    mlir::cf::ControlFlowDialect>();
  }

  auto runOnOperation() -> void final {
    auto& context = getContext();

    mlir::ConversionTarget target(context);

    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LowerToLLVMOptions options{&context};

    options.useBarePtrCallConv = true;

    mlir::RewritePatternSet patterns{&context};
    mlir::LLVMTypeConverter type_converter{&context, options};

    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(type_converter,
                                                       patterns);
    mlir::populateMathToLLVMConversionPatterns(type_converter, patterns);

    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(type_converter,
                                                          patterns);

    mlir::populateFinalizeMemRefToLLVMConversionPatterns(type_converter,
                                                         patterns);
    mlir::populateFuncToLLVMConversionPatterns(type_converter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

export auto createLlvmLoweringPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<AxonToLlvmLoweringPass>();
}

}  // namespace axon
