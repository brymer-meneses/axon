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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

export module axon.mlir:lowering;

namespace axon {

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : mlir::OpConversionPattern<BinaryOp> {
  using mlir::OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

  auto matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto lhs_tensor = llvm::cast<mlir::TensorType>(adaptor.getLhs().getType());
    auto rhs_tensor = llvm::cast<mlir::TensorType>(adaptor.getRhs().getType());

    if (rhs_tensor.getShape().equals(lhs_tensor.getShape())) {
      auto new_op = LoweredBinaryOp::create(rewriter, loc, adaptor.getLhs(),
                                            adaptor.getRhs());
      rewriter.replaceOp(op, new_op);
      return mlir::success();
    }

    std::unreachable();
  }
};

struct GetDataOpLowering : mlir::OpConversionPattern<GetDataOp> {
  using mlir::OpConversionPattern<GetDataOp>::OpConversionPattern;

  auto matchAndRewrite(GetDataOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto tensor_ref_type = mlir::cast<TensorRefType>(op.getInput().getType());
    auto tensor_type = mlir::RankedTensorType::get(
        tensor_ref_type.getShape(), tensor_ref_type.getElementType());

    auto memref =
        TupleAccessOp::create(rewriter, loc, adaptor.getInput(), 0).getResult();
    auto new_op = mlir::bufferization::ToTensorOp::create(
        rewriter, loc, tensor_type, memref, /*restrict=*/true,
        /*writable=*/false);
    rewriter.replaceOp(op, new_op);
    return mlir::success();
  }
};

struct AccumulateGradOpLowering : mlir::OpConversionPattern<AccumulateGradOp> {
  using mlir::OpConversionPattern<AccumulateGradOp>::OpConversionPattern;

  auto matchAndRewrite(AccumulateGradOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto tensor_type =
        llvm::dyn_cast<TensorRefType>(op.getAccumulator().getType());

    // Since  `adaptor.getAccumulator` must require a gradient and because
    // `TensorRefType` is lowered to tuple<memref, memref>, then to access the
    // grad memref we need to invoke the tuple access op to access the gradient.
    auto grad_ref =
        TupleAccessOp::create(rewriter, loc, adaptor.getAccumulator(), 1);
    auto memref_type = mlir::MemRefType::get(tensor_type.getShape(),
                                             tensor_type.getElementType());
    auto value_as_memref = mlir::bufferization::ToBufferOp::create(
        rewriter, loc, memref_type, adaptor.getValue());

    mlir::linalg::AddOp::create(rewriter, loc,
                                mlir::ValueRange{grad_ref, value_as_memref},
                                mlir::ValueRange{grad_ref});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct FillLikeOpLowering : mlir::OpConversionPattern<FillLikeOp> {
  using mlir::OpConversionPattern<FillLikeOp>::OpConversionPattern;

  auto matchAndRewrite(FillLikeOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto tensor = mlir::cast<mlir::TensorType>(op.getTensor().getType());

    auto result_buffer = mlir::tensor::EmptyOp::create(
        rewriter, loc, tensor.getShape(), tensor.getElementType());

    auto fill_value = rewriter.create<mlir::arith::ConstantFloatOp>(
        loc, rewriter.getF64Type(), op.getFillValue());

    auto fill_op = mlir::linalg::FillOp::create(
        rewriter, loc, mlir::ValueRange{fill_value},
        mlir::ValueRange{result_buffer});

    rewriter.replaceOp(op, fill_op.getResult(0));
    return mlir::success();
  }
};

struct AxonToStdTypeConverter : mlir::TypeConverter {
  AxonToStdTypeConverter(mlir::MLIRContext* ctx) {
    addConversion([](mlir::Type type) -> mlir::Type { return type; });
    addConversion([ctx](TensorRefType tensor_ref) -> mlir::Type {
      auto element_type = tensor_ref.getElementType();
      auto shape = tensor_ref.getShape();
      auto memref_type = mlir::MemRefType::get(shape, element_type);
      if (tensor_ref.getRequiresGrad()) {
        return mlir::TupleType::get(ctx,
                                    mlir::TypeRange{memref_type, memref_type});
      }
      return memref_type;
    });
  }
};

using AddOpLowering = BinaryOpLowering<AddOp, mlir::arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<MulOp, mlir::arith::MulFOp>;

struct AxonToStandardLoweringPass
    : mlir::PassWrapper<AxonToStandardLoweringPass,
                        mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AxonToStandardLoweringPass)

  auto getArgument() const -> llvm::StringRef override {
    return "axon-to-standard-lowering";
  }

  auto getDependentDialects(mlir::DialectRegistry& registry) const
      -> void override {
    registry
        .insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                mlir::bufferization::BufferizationDialect, mlir::BuiltinDialect,
                mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();
    // TODO: this shouldn't be here
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  }

  auto runOnOperation() -> void final {
    auto& context = getContext();

    mlir::ConversionTarget target(context);

    target
        .addLegalDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                         mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                         mlir::bufferization::BufferizationDialect,
                         mlir::BuiltinDialect, mlir::tensor::TensorDialect>();
    target.addLegalOp<TupleAccessOp>();
    target
        .addIllegalOp<AddOp, MulOp, GetDataOp, AccumulateGradOp, FillLikeOp>();

    AxonToStdTypeConverter type_converter{&context};

    mlir::RewritePatternSet patterns{&context};
    patterns.add<AddOpLowering, MulOpLowering, GetDataOpLowering,
                 AccumulateGradOpLowering, FillLikeOpLowering>(type_converter,
                                                               &context);

    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, type_converter);
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return type_converter.isSignatureLegal(op.getFunctionType()) &&
                 type_converter.isLegal(&op.getBody());
        });

    mlir::populateReturnOpTypeConversionPattern(patterns, type_converter);
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) { return type_converter.isLegal(op); });

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct AxonToLlvmTypeConverter : mlir::LLVMTypeConverter {
  AxonToLlvmTypeConverter(mlir::MLIRContext* ctx,
                          const mlir::LowerToLLVMOptions& options)
      : mlir::LLVMTypeConverter(ctx, options) {
    addConversion([ctx, this](mlir::TupleType tuple_type) -> mlir::Type {
      AXON_DCHECK(getOptions().useBarePtrCallConv, "");

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
    options.useBarePtrCallConv = true;

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

export auto createStandardLoweringPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<AxonToStandardLoweringPass>();
}

}  // namespace axon
