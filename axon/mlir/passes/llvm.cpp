module;

#include <memory>

#include "axon/base/dcheck.h"
#include "axon/mlir/dialect/dialect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
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

struct AxonToLlvmTypeConverter : mlir::LLVMTypeConverter {
  AxonToLlvmTypeConverter(mlir::MLIRContext* ctx,
                          const mlir::LowerToLLVMOptions& options)
      : mlir::LLVMTypeConverter(ctx, options) {
    addConversion([ctx, this](mlir::TupleType tuple_type) -> mlir::Type {
      llvm::SmallVector<mlir::Type> llvm_types;
      for (mlir::Type elem_type : tuple_type.getTypes()) {
        mlir::Type converted_type = convertType(elem_type);
        if (!converted_type) {
          return {};
        }
        if (!mlir::LLVM::isCompatibleType(converted_type)) {
          return {};
        }

        llvm_types.push_back(converted_type);
      }

      return mlir::LLVM::LLVMStructType::getLiteral(ctx, llvm_types);
    });
  }
};

struct TupleAccessOpLowering : mlir::OpConversionPattern<TupleAccessOp> {
  using mlir::OpConversionPattern<TupleAccessOp>::OpConversionPattern;

  auto matchAndRewrite(TupleAccessOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto input = adaptor.getInput();
    auto index = adaptor.getIndex();

    auto new_op =
        mlir::LLVM::ExtractValueOp::create(rewriter, loc, input, index);
    rewriter.replaceOp(op, new_op);
    return mlir::success();
  }
};

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

    // All shapes are known at compile time so memrefs should be lowered to a
    // pointer.
    options.useBarePtrCallConv = false;

    mlir::RewritePatternSet patterns{&context};
    AxonToLlvmTypeConverter type_converter{&context, options};

    patterns.add<TupleAccessOpLowering>(type_converter, &context);

    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(type_converter,
                                                       patterns);
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
