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
struct ElementWiseBinaryOpLowering : mlir::OpConversionPattern<BinaryOp> {
  using mlir::OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

  auto matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto tensor_type = mlir::cast<mlir::TensorType>(op.getResult().getType());
    auto empty = mlir::tensor::EmptyOp::create(
        rewriter, loc, tensor_type.getShape(), tensor_type.getElementType());

    auto new_op = LoweredBinaryOp::create(
        rewriter, loc, mlir::ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        mlir::ValueRange{empty});

    rewriter.replaceOp(op, new_op.getResult(0));
    return mlir::success();
  }
};

using AddOpLowering = ElementWiseBinaryOpLowering<AddOp, mlir::linalg::AddOp>;
using MulOpLowering = ElementWiseBinaryOpLowering<MulOp, mlir::linalg::MulOp>;

struct MatMulOpLowering : mlir::OpConversionPattern<MatMulOp> {
  using mlir::OpConversionPattern<MatMulOp>::OpConversionPattern;

  auto matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto result_tensor_type =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto rank = result_tensor_type.getRank();

    if (rank > 3 || rank <= 1) {
      op.emitError("Got invalid rank for MatMulOp.");
      return mlir::failure();
    }

    auto empty_op = mlir::tensor::EmptyOp::create(
        rewriter, loc, result_tensor_type.getShape(),
        result_tensor_type.getElementType());

    if (result_tensor_type.getRank() == 3) {
      auto new_op = mlir::linalg::BatchMatmulOp::create(
          rewriter, loc, mlir::ValueRange{adaptor.getLhs(), adaptor.getRhs()},
          mlir::ValueRange{empty_op});
      rewriter.replaceOp(op, new_op.getResult(0));
      return mlir::success();
    }

    if (result_tensor_type.getRank() == 2) {
      auto new_op = mlir::linalg::MatmulOp::create(
          rewriter, loc, mlir::ValueRange{adaptor.getLhs(), adaptor.getRhs()},
          mlir::ValueRange{empty_op});
      rewriter.replaceOp(op, new_op.getResult(0));
      return mlir::success();
    }

    std::unreachable();
  }
};

struct BroadcastOpLowering : mlir::OpConversionPattern<BroadcastOp> {
  using mlir::OpConversionPattern<BroadcastOp>::OpConversionPattern;

  auto matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto result_tensor_type =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());

    auto result_element_type = result_tensor_type.getElementType();
    auto result_shape = result_tensor_type.getShape();

    auto init_op = mlir::tensor::EmptyOp::create(
        rewriter, op.getLoc(), result_shape, result_element_type);

    llvm::SmallVector<int64_t> dimensions;
    for (const auto& expansion : op.getExpansions()) {
      auto pair = mlir::cast<mlir::ArrayAttr>(expansion);

      auto dim_attr = mlir::cast<mlir::IntegerAttr>(pair[0]);
      dimensions.push_back(dim_attr.getInt());
    }

    auto operand = adaptor.getOperand();
    auto new_op = mlir::linalg::BroadcastOp::create(
        rewriter, op.getLoc(), operand, init_op, dimensions);

    rewriter.replaceOp(op, new_op);
    return mlir::success();
  }
};

struct SumOpLowering : mlir::OpConversionPattern<SumOp> {
  using mlir::OpConversionPattern<SumOp>::OpConversionPattern;

  auto matchAndRewrite(SumOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    if (op.getKeepDims()) {
    }
  }
};

struct SqueezeOpLowering : mlir::OpConversionPattern<SqueezeOp> {
  using mlir::OpConversionPattern<SqueezeOp>::OpConversionPattern;

  auto matchAndRewrite(SqueezeOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {}
};

struct UmsqueezeOpLowering : mlir::OpConversionPattern<UnsqueezeOp> {
  using mlir::OpConversionPattern<UnsqueezeOp>::OpConversionPattern;

  auto matchAndRewrite(UnsqueezeOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {}
};

struct ReshapeOpLowering : mlir::OpConversionPattern<ReshapeOp> {
  using mlir::OpConversionPattern<ReshapeOp>::OpConversionPattern;

  auto matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto operand = op.getOperand();
    auto element_type =
        mlir::cast<mlir::RankedTensorType>(operand.getType()).getElementType();

    auto result_tensor =
        mlir::RankedTensorType::get(op.getTargetShape(), element_type);
    auto source_tensor = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto reassociation_indices =
        mlir::getReassociationIndicesForReshape(source_tensor, result_tensor);

    if (!reassociation_indices) {
      op.emitError("Shapes cannot be reshaped.");
      return mlir::failure();
    }

    auto rehape_op = mlir::tensor::ExpandShapeOp::create(
        rewriter, op.getLoc(), result_tensor, operand, *reassociation_indices);

    rewriter.replaceOp(op, rehape_op);
    return mlir::success();
  }
};

struct GetDataOpLowering : mlir::OpConversionPattern<GetDataOp> {
  using mlir::OpConversionPattern<GetDataOp>::OpConversionPattern;

  auto matchAndRewrite(GetDataOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto tensor_ref = mlir::cast<TensorRefType>(op.getInput().getType());

    auto memref =
        TupleAccessOp::create(rewriter, loc, adaptor.getInput(), 0).getResult();
    auto new_op = mlir::bufferization::ToTensorOp::create(
        rewriter, loc, tensor_ref.getTensorType(), memref,
        /*restrict=*/true,
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

    // Since `adaptor.getAccumulator` must require a gradient and because
    // `TensorRefType` is lowered to tuple<memref, memref>, then to access the
    // grad memref we need to invoke the tuple access op to access the gradient.
    auto grad_ref =
        TupleAccessOp::create(rewriter, loc, adaptor.getAccumulator(), 1);
    auto value_as_memref = mlir::bufferization::ToBufferOp::create(
        rewriter, loc, tensor_type.getMemRefType(), adaptor.getValue());

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
    target.addIllegalOp<GetDataOp, AccumulateGradOp, FillLikeOp, AddOp, MulOp,
                        MatMulOp, BroadcastOp>();

    AxonToStandardTypeConverter type_converter{&context};

    mlir::RewritePatternSet patterns{&context};
    patterns.add<GetDataOpLowering, AccumulateGradOpLowering,
                 FillLikeOpLowering, AddOpLowering, MulOpLowering,
                 MatMulOpLowering, BroadcastOpLowering>(type_converter,
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
