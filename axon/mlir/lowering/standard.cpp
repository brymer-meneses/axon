module;

#include <algorithm>

#include "axon/mlir/dialect/dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
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
    auto result_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());

    llvm::SmallVector<int64_t> dimensions_to_expand;
    for (auto [i, attr] : llvm::enumerate(op.getExpansions())) {
      auto pair = mlir::cast<mlir::ArrayAttr>(attr);
      auto dim = mlir::cast<mlir::IntegerAttr>(pair[0]).getInt();
      dimensions_to_expand.push_back(dim);
    }

    llvm::SmallVector<mlir::AffineExpr> affine_exprs;
    for (auto i = 0; i < result_tensor.getRank(); i += 1) {
      if (std::ranges::contains(dimensions_to_expand, i)) {
        affine_exprs.push_back(mlir::getAffineConstantExpr(0, op.getContext()));
      } else {
        affine_exprs.push_back(mlir::getAffineDimExpr(i, op.getContext()));
      }
    }
    auto input_map =
        mlir::AffineMap::get(result_tensor.getRank(),
                             /*symbolCount=*/0, affine_exprs, op.getContext());
    auto output_map = mlir::AffineMap::getMultiDimIdentityMap(
        result_tensor.getRank(), op.getContext());

    llvm::SmallVector<mlir::AffineMap> indexing_maps = {input_map, output_map};
    llvm::SmallVector iterator_types(result_tensor.getRank(),
                                     mlir::utils::IteratorType::parallel);

    auto init_op = mlir::tensor::EmptyOp::create(
        rewriter, op.getLoc(), result_tensor.getShape(),
        result_tensor.getElementType());

    auto broadcast_op = mlir::linalg::GenericOp::create(
        rewriter, op.getLoc(),
        /*resultTensorTypes=*/mlir::TypeRange{op.getResult().getType()},
        /*inputs=*/mlir::ValueRange{adaptor.getOperand()},
        /*outputs=*/mlir::ValueRange{init_op},

        indexing_maps, iterator_types,
        [&](mlir::OpBuilder& builder, mlir::Location loc,
            mlir::ValueRange args) {
          mlir::linalg::YieldOp::create(builder, loc, args[0]);
        });

    rewriter.replaceOp(op, broadcast_op.getResult(0));
    return mlir::success();
  }
};

struct SumOpLowering : mlir::OpConversionPattern<SumOp> {
  using mlir::OpConversionPattern<SumOp>::OpConversionPattern;

  auto matchAndRewrite(SumOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto input_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
    auto result_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());

    auto input_rank = input_tensor.getRank();
    auto dim_to_reduce = static_cast<int64_t>(op.getDim());

    // If we have keepDims on then we need to set `0` as the accumulator for
    // that dimension.
    mlir::AffineMap output_map;
    if (op.getKeepDims()) {
      llvm::SmallVector<mlir::AffineExpr> affine_exprs;
      for (int64_t i = 0; i < input_rank; i += 1) {
        if (i == dim_to_reduce) {
          affine_exprs.push_back(
              mlir::getAffineConstantExpr(0, op.getContext()));
        } else {
          affine_exprs.push_back(mlir::getAffineDimExpr(i, op.getContext()));
        }
      }
      output_map = mlir::AffineMap::get(input_rank, /*symbolCount=*/0,
                                        affine_exprs, op.getContext());
    } else {
      llvm::SmallVector<mlir::AffineExpr> affine_exprs;
      for (int64_t i = 0; i < input_rank; i += 1) {
        if (i != dim_to_reduce) {
          affine_exprs.push_back(mlir::getAffineDimExpr(i, op.getContext()));
        }
      }
      output_map = mlir::AffineMap::get(input_rank, /*symbolCount=*/0,
                                        affine_exprs, op.getContext());
    }

    auto input_map = mlir::AffineMap::getMultiDimIdentityMap(
        result_tensor.getRank(), op.getContext());

    llvm::SmallVector<mlir::AffineMap> indexing_maps = {input_map, output_map};

    llvm::SmallVector iterator_types(input_rank,
                                     mlir::utils::IteratorType::parallel);
    iterator_types[dim_to_reduce] = mlir::utils::IteratorType::reduction;

    auto init_op =
        mlir::tensor::EmptyOp::create(rewriter, loc, result_tensor.getShape(),
                                      result_tensor.getElementType());

    auto sum_op = mlir::linalg::GenericOp::create(
        rewriter, loc,
        /*resultTensorTypes=*/mlir::TypeRange{op.getResult().getType()},

        /*inputs=*/mlir::ValueRange{adaptor.getOperand()},
        /*outputs=*/mlir::ValueRange{init_op},

        indexing_maps, iterator_types,
        [&](mlir::OpBuilder& builder, mlir::Location loc,
            mlir::ValueRange args) {
          auto add =
              mlir::arith::AddFOp::create(builder, loc, args[0], args[1]);
          mlir::linalg::YieldOp::create(builder, loc, add.getResult());
        });

    rewriter.replaceOp(op, sum_op.getResult(0));
    return mlir::success();
  }
};

struct SqueezeOpLowering : mlir::OpConversionPattern<SqueezeOp> {
  using mlir::OpConversionPattern<SqueezeOp>::OpConversionPattern;

  auto matchAndRewrite(SqueezeOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto input_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
    auto result_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());

    auto reassociation_indices =
        mlir::getReassociationIndicesForReshape(input_tensor, result_tensor);
    if (!reassociation_indices) {
      op.emitError(
          std::format("Failed to compute reassociation indices for {} and {}",
                      input_tensor.getShape(), result_tensor.getShape()));
      return mlir::failure();
    }

    auto collapse_op = mlir::tensor::CollapseShapeOp::create(
        rewriter, op.getLoc(), adaptor.getOperand(), *reassociation_indices);
    rewriter.replaceOp(op, collapse_op);
    return mlir::success();
  }
};

struct UnqueezeOpLowering : mlir::OpConversionPattern<UnsqueezeOp> {
  using mlir::OpConversionPattern<UnsqueezeOp>::OpConversionPattern;

  auto matchAndRewrite(UnsqueezeOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto input_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
    auto result_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());

    auto reassociation_indices =
        mlir::getReassociationIndicesForReshape(input_tensor, result_tensor);
    if (!reassociation_indices) {
      op.emitError(
          std::format("Failed to compute reassociation indices for {} and {}",
                      input_tensor.getShape(), result_tensor.getShape()));
      return mlir::failure();
    }

    auto expand_op = mlir::tensor::ExpandShapeOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), adaptor.getOperand(),
        *reassociation_indices);
    rewriter.replaceOp(op, expand_op);
    return mlir::success();
  }
};

struct ReshapeOpLowering : mlir::OpConversionPattern<ReshapeOp> {
  using mlir::OpConversionPattern<ReshapeOp>::OpConversionPattern;

  auto matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto result_type = op.getResult().getType();
    auto result_tensor = mlir::cast<mlir::RankedTensorType>(result_type);
    auto input_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());
    auto reassociation_indices =
        mlir::getReassociationIndicesForReshape(input_tensor, result_tensor);
    if (!reassociation_indices) {
      op.emitError(
          std::format("Failed to compute reassociation indices for {} and {}",
                      input_tensor.getShape(), result_tensor.getShape()));
      return mlir::failure();
    }

    if (input_tensor.getRank() < result_tensor.getRank()) {
      auto expand_op = mlir::tensor::ExpandShapeOp::create(
          rewriter, op.getLoc(), op.getResult().getType(), adaptor.getOperand(),
          *reassociation_indices);
      rewriter.replaceOp(op, expand_op);
      return mlir::success();
    }

    if (input_tensor.getRank() > result_tensor.getRank()) {
      auto expand_op = mlir::tensor::CollapseShapeOp::create(
          rewriter, op.getLoc(), op.getResult().getType(), adaptor.getOperand(),
          *reassociation_indices);
      rewriter.replaceOp(op, expand_op);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct TransposeOpLowering : mlir::OpConversionPattern<TransposeOp> {
  using mlir::OpConversionPattern<TransposeOp>::OpConversionPattern;

  auto matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto result_type = op.getResult().getType();
    auto result_tensor = mlir::cast<mlir::RankedTensorType>(result_type);

    llvm::SmallVector<uint32_t> targets;
    for (auto i = 0; i < result_tensor.getRank(); i += 1) {
      targets.push_back(i);
    }
    std::swap(targets[op.getFrom()], targets[op.getTo()]);

    auto input_map = mlir::AffineMap::getMultiDimIdentityMap(
        result_tensor.getRank(), op.getContext());
    auto output_map = mlir::AffineMap::getMultiDimMapWithTargets(
        result_tensor.getRank(), targets, op.getContext());

    llvm::SmallVector iterator_types(result_tensor.getRank(),
                                     mlir::utils::IteratorType::parallel);
    llvm::SmallVector indexing_maps = {input_map, output_map};

    auto init_op =
        mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), result_type, {});

    auto transpose_op = mlir::linalg::GenericOp::create(
        rewriter, op.getLoc(), mlir::TypeRange{result_type},
        mlir::ValueRange{adaptor.getOperand()}, mlir::ValueRange{init_op},
        indexing_maps, iterator_types,
        [&](mlir::OpBuilder& builder, mlir::Location loc,
            mlir::ValueRange args) {
          mlir::linalg::YieldOp::create(builder, loc, args[0]);
        });

    rewriter.replaceOp(op, transpose_op.getResult(0));
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

    // Since `adaptor.getAccumulator` must require a gradient (which ensures
    // that this is a valid tuple access) and because `TensorRefType` is lowered
    // to tuple<memref, memref>, then to access the grad memref we need to
    // invoke a tuple access op to access it.
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
                        MatMulOp, SumOp, BroadcastOp, TransposeOp, SqueezeOp,
                        UnsqueezeOp, ReshapeOp>();

    AxonToStandardTypeConverter type_converter{&context};

    mlir::RewritePatternSet patterns{&context};
    patterns
        .add<GetDataOpLowering, AccumulateGradOpLowering, FillLikeOpLowering,
             AddOpLowering, MulOpLowering, MatMulOpLowering, SumOpLowering,
             BroadcastOpLowering, TransposeOpLowering, SqueezeOpLowering,
             UnqueezeOpLowering, ReshapeOpLowering>(type_converter, &context);

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
