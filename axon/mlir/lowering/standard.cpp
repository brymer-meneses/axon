module;

#include "axon/mlir/dialect/dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

export module axon.mlir:standard_lowering;

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

struct AxonToStandardTypeConverter : mlir::TypeConverter {
  AxonToStandardTypeConverter(mlir::MLIRContext* ctx) {
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

    AxonToStandardTypeConverter type_converter{&context};

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

export auto createStandardLoweringPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<AxonToStandardLoweringPass>();
}

}  // namespace axon
