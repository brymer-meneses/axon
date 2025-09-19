module;
#include <algorithm>
#include <optional>
#include <utility>

#include "axon/base/macros.h"
#include "axon/mlir/dialect/dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

export module axon.mlir.passes:standard_lowering;

namespace axon {

static auto createZerosLike(mlir::ConversionPatternRewriter& rewriter,
                            mlir::TensorType tensor_type, mlir::Location loc)
    -> mlir::Value {
  auto element_type = tensor_type.getElementType();
  auto empty_op = mlir::tensor::EmptyOp::create(
      rewriter, loc, tensor_type.getShape(), element_type);
  auto fill_value = mlir::arith::ConstantOp::create(
      rewriter, loc, element_type, rewriter.getZeroAttr(element_type));
  auto fill_op =
      mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{fill_value},
                                   mlir::ValueRange{empty_op})
          .getResult(0);
  return fill_op;
}

template <typename BinaryOp, typename LoweredBinaryOp>
struct ElementWiseBinaryOpLowering : mlir::OpConversionPattern<BinaryOp> {
  using mlir::OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

  auto matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto tensor_type =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto empty_op = createZerosLike(rewriter, tensor_type, loc);

    auto new_op = LoweredBinaryOp::create(
        rewriter, loc, mlir::ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        mlir::ValueRange{empty_op});

    rewriter.replaceOp(op, new_op.getResult(0));
    return mlir::success();
  }
};

using AddOpLowering = ElementWiseBinaryOpLowering<AddOp, mlir::linalg::AddOp>;
using MulOpLowering = ElementWiseBinaryOpLowering<MulOp, mlir::linalg::MulOp>;
using SubOpLowering = ElementWiseBinaryOpLowering<SubOp, mlir::linalg::SubOp>;

struct MatMulOpLowering : mlir::OpConversionPattern<MatMulOp> {
  using mlir::OpConversionPattern<MatMulOp>::OpConversionPattern;

  auto matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto result_tensor_type = op.getResult().getType();
    auto rank = result_tensor_type.getRank();

    if (rank > 3 || rank <= 1) {
      return rewriter.notifyMatchFailure(op, "Got invalid rank for MatMulOp");
    }

    auto init_op = createZerosLike(rewriter, result_tensor_type, loc);
    auto indexing_maps = getIndexingMaps(op, rewriter);

    if (result_tensor_type.getRank() == 3) {
      auto new_op = mlir::linalg::BatchMatmulOp::create(
          rewriter, loc,
          /*inputs=*/mlir::ValueRange{adaptor.getLhs(), adaptor.getRhs()},
          /*outputs=*/mlir::ValueRange{init_op},
          /*attributes=*/{{"indexing_maps", indexing_maps}});
      rewriter.replaceOp(op, new_op.getResult(0));
      return mlir::success();
    }

    if (result_tensor_type.getRank() == 2) {
      auto new_op = mlir::linalg::MatmulOp::create(
          rewriter, loc,
          /*inputs=*/mlir::ValueRange{adaptor.getLhs(), adaptor.getRhs()},
          /*outputs=*/mlir::ValueRange{init_op},
          /*attributes=*/{{"indexing_maps", indexing_maps}});

      rewriter.replaceOp(op, new_op.getResult(0));
      return mlir::success();
    }

    AXON_UNREACHABLE("MatMul should only have ranks of 2 or 3");
  }

  static auto getIndexingMaps(MatMulOp op,
                              mlir::ConversionPatternRewriter& rewriter)
      -> mlir::ArrayAttr {
    auto context = rewriter.getContext();
    auto rank = op.getResult().getType().getRank();

    if (rank == 2) {
      auto m = mlir::getAffineDimExpr(0, context);
      auto n = mlir::getAffineDimExpr(1, context);
      auto k = mlir::getAffineDimExpr(2, context);

      llvm::SmallVector lhs_exprs =
          op.getTransposeLhs() ? llvm::SmallVector<mlir::AffineExpr>{k, m}
                               : llvm::SmallVector<mlir::AffineExpr>{m, k};
      auto rhs_exprs = op.getTransposeRhs()
                           ? llvm::SmallVector<mlir::AffineExpr>{n, k}
                           : llvm::SmallVector<mlir::AffineExpr>{k, n};

      for (auto dim : llvm::reverse(op.getExpandedLhsDims())) {
        lhs_exprs.erase(lhs_exprs.begin() + dim);
      }
      for (auto dim : llvm::reverse(op.getExpandedRhsDims())) {
        rhs_exprs.erase(rhs_exprs.begin() + dim);
      }

      auto result_map = mlir::AffineMap::get(3, 0, {m, n}, context);
      auto lhs_map = mlir::AffineMap::get(3, 0, lhs_exprs, context);
      auto rhs_map = mlir::AffineMap::get(3, 0, rhs_exprs, context);

      return rewriter.getAffineMapArrayAttr({lhs_map, rhs_map, result_map});
    }

    if (rank == 3) {
      auto b = mlir::getAffineDimExpr(0, context);
      auto m = mlir::getAffineDimExpr(1, context);
      auto n = mlir::getAffineDimExpr(2, context);
      auto k = mlir::getAffineDimExpr(3, context);

      auto lhs_exprs = op.getTransposeLhs()
                           ? llvm::SmallVector<mlir::AffineExpr>{b, k, m}
                           : llvm::SmallVector<mlir::AffineExpr>{b, m, k};

      auto rhs_exprs = op.getTransposeRhs()
                           ? llvm::SmallVector<mlir::AffineExpr>{b, n, k}
                           : llvm::SmallVector<mlir::AffineExpr>{b, k, n};

      for (auto dim : llvm::reverse(op.getExpandedLhsDims())) {
        lhs_exprs.erase(lhs_exprs.begin() + dim);
      }
      for (auto dim : llvm::reverse(op.getExpandedRhsDims())) {
        rhs_exprs.erase(rhs_exprs.begin() + dim);
      }

      auto result_map = mlir::AffineMap::get(4, 0, {b, m, n}, context);
      auto lhs_map = mlir::AffineMap::get(4, 0, lhs_exprs, context);
      auto rhs_map = mlir::AffineMap::get(4, 0, rhs_exprs, context);

      return rewriter.getAffineMapArrayAttr({lhs_map, rhs_map, result_map});
    }

    AXON_UNREACHABLE("MatMul should only have ranks of 2 or 3");
  }
};

struct ExpandDimsOpLowering : mlir::OpConversionPattern<ExpandDimsOp> {
  using mlir::OpConversionPattern<ExpandDimsOp>::OpConversionPattern;

  auto matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto result_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());

    llvm::SmallVector<int64_t> dimensions_to_expand;
    for (auto attr : op.getMappings()) {
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

    auto init_op = createZerosLike(rewriter, result_tensor, loc);

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

    auto init_op = createZerosLike(rewriter, result_tensor, loc);

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
      return rewriter.notifyMatchFailure(
          op,
          std::format("Failed to compute reassociation indices for {} -> {}",
                      input_tensor.getShape(), result_tensor.getShape()));
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
      return rewriter.notifyMatchFailure(
          op,
          std::format("Failed to compute reassociation indices for {} and {}",
                      input_tensor.getShape(), result_tensor.getShape()));
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
      return rewriter.notifyMatchFailure(
          op,
          std::format("Failed to compute reassociation indices for {} and {}",
                      input_tensor.getShape(), result_tensor.getShape()));
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

    auto init_op = adaptor.getOperand();

    auto transpose_op = mlir::linalg::GenericOp::create(
        rewriter, op.getLoc(), mlir::TypeRange{result_type},
        mlir::ValueRange{adaptor.getOperand()}, mlir::ValueRange{init_op},
        indexing_maps, iterator_types,
        [](mlir::OpBuilder& builder, mlir::Location loc,
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

struct GetGradOpLowering : mlir::OpConversionPattern<GetGradOp> {
  using mlir::OpConversionPattern<GetGradOp>::OpConversionPattern;

  auto matchAndRewrite(GetGradOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto tensor_ref = mlir::cast<TensorRefType>(op.getInput().getType());
    auto memref =
        TupleAccessOp::create(rewriter, loc, adaptor.getInput(), 1).getResult();

    auto new_op = mlir::bufferization::ToTensorOp::create(
        rewriter, loc, tensor_ref.getTensorType(), memref,
        /*restrict=*/true,
        /*writable=*/true);

    rewriter.replaceOp(op, new_op);
    return mlir::success();
  }
};

struct AccumulateOpLowering : mlir::OpConversionPattern<AccumulateOp> {
  using mlir::OpConversionPattern<AccumulateOp>::OpConversionPattern;

  auto matchAndRewrite(AccumulateOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto sink = adaptor.getSink();
    auto sink_tensor_type =
        mlir::cast<mlir::RankedTensorType>(adaptor.getSink().getType());

    llvm::SmallVector iterator_types(sink_tensor_type.getRank(),
                                     mlir::utils::IteratorType::parallel);
    llvm::SmallVector indexing_maps(
        3, mlir::AffineMap::getMultiDimIdentityMap(sink_tensor_type.getRank(),
                                                   op.getContext()));
    auto source = adaptor.getSource();
    auto accumulation_op =
        mlir::linalg::GenericOp::create(
            rewriter, loc, mlir::TypeRange{sink_tensor_type},
            mlir::ValueRange{sink, source}, mlir::ValueRange{sink},
            indexing_maps, iterator_types,
            [](mlir::OpBuilder& builder, mlir::Location loc,
               mlir::ValueRange args) {
              auto addition =
                  mlir::arith::AddFOp::create(builder, loc, args[0], args[1]);
              mlir::linalg::YieldOp::create(builder, loc, addition.getResult());
            })
            .getResult(0);

    auto materialize_op =
        mlir::bufferization::MaterializeInDestinationOp::create(
            rewriter, loc, accumulation_op, sink);

    rewriter.replaceOp(op, materialize_op.getResult());
    return mlir::success();
  }
};

struct FillOpLowering : mlir::OpConversionPattern<FillOp> {
  using mlir::OpConversionPattern<FillOp>::OpConversionPattern;

  auto matchAndRewrite(FillOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto tensor = mlir::cast<mlir::TensorType>(op.getResult().getType());

    if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(op.getFillValue())) {
      auto fill_value =
          mlir::arith::ConstantOp::create(rewriter, loc, float_attr);
      auto fill_op =
          mlir::tensor::SplatOp::create(rewriter, loc, fill_value, tensor);
      rewriter.replaceOp(op, fill_op);
      return mlir::success();
    }

    if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(op.getFillValue())) {
      auto fill_value =
          mlir::arith::ConstantOp::create(rewriter, loc, int_attr);
      auto fill_op =
          mlir::tensor::SplatOp::create(rewriter, loc, fill_value, tensor);
      rewriter.replaceOp(op, fill_op);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct NegOpLowering : mlir::OpConversionPattern<NegOp> {
  using mlir::OpConversionPattern<NegOp>::OpConversionPattern;

  auto matchAndRewrite(NegOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto tensor = mlir::cast<mlir::RankedTensorType>(op.getOperand().getType());

    auto init_op = createZerosLike(rewriter, tensor, loc);

    auto neg_op = mlir::linalg::NegFOp::create(
        rewriter, loc, mlir::ValueRange{adaptor.getOperand()},
        mlir::ValueRange{init_op});

    rewriter.replaceOp(op, neg_op.getResult(0));
    return mlir::success();
  }
};

struct ScalarMulOpLowering : mlir::OpConversionPattern<ScalarMulOp> {
  using mlir::OpConversionPattern<ScalarMulOp>::OpConversionPattern;

  auto matchAndRewrite(ScalarMulOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto result_type = op.getOperand().getType();

    if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(op.getScalar())) {
      auto scalar_constant =
          mlir::arith::ConstantOp::create(rewriter, loc, float_attr)
              .getResult();
      auto scalar_tensor = mlir::tensor::SplatOp::create(
          rewriter, loc, scalar_constant, result_type);

      auto empty_op =
          mlir::tensor::EmptyOp::create(rewriter, loc, result_type, {});
      auto prod = mlir::linalg::MulOp::create(
          rewriter, loc, mlir::ValueRange{scalar_tensor, adaptor.getOperand()},
          mlir::ValueRange{empty_op});

      rewriter.replaceOp(op, prod.getResult(0));
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct ConstantOpLowering : mlir::OpConversionPattern<ConstantOp> {
  using mlir::OpConversionPattern<ConstantOp>::OpConversionPattern;

  auto matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto constant_value = op.getValue();
    if (constant_value.isSplat()) {
      return handleSplatLowering(op, constant_value, rewriter);
    }
    return rewriter.notifyMatchFailure(
        op.getLoc(), "Non-splat constant tensors are not supported yet");
  }

  auto handleSplatLowering(ConstantOp op, mlir::ElementsAttr constant_value,
                           mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult {
    auto element_type = op.getValue().getElementType();
    auto loc = op.getLoc();
    if (element_type.isFloat()) {
      auto value = constant_value.getSplatValue<llvm::APFloat>();
      auto scalar_constant = mlir::arith::ConstantFloatOp::create(
          rewriter, op.getLoc(), rewriter.getFloatAttr(element_type, value));

      auto splat = mlir::tensor::SplatOp::create(rewriter, loc, scalar_constant,
                                                 op.getResult().getType());

      rewriter.replaceOp(op, splat);
      return mlir::success();
    } else if (element_type.isInteger()) {
      auto value = constant_value.getSplatValue<llvm::APInt>();
      auto scalar_constant = mlir::arith::ConstantIntOp::create(
          rewriter, op.getLoc(), rewriter.getIntegerAttr(element_type, value));

      auto splat = mlir::tensor::SplatOp::create(rewriter, loc, scalar_constant,
                                                 op.getResult().getType());

      rewriter.replaceOp(op, splat);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct PowOpLowering : mlir::OpConversionPattern<PowOp> {
  using mlir::OpConversionPattern<PowOp>::OpConversionPattern;

  auto matchAndRewrite(PowOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto result_tensor_type = op.getResult().getType();

    auto exponent_constant = mlir::arith::ConstantFloatOp::create(
        rewriter, loc,
        rewriter.getF64FloatAttr(op.getExponent().convertToDouble()));

    auto exponent_tensor = mlir::tensor::SplatOp::create(
        rewriter, loc, exponent_constant, result_tensor_type.getShape());

    auto pow_op = mlir::math::PowFOp::create(
        rewriter, loc, adaptor.getOperand(), exponent_tensor);

    rewriter.replaceOp(op, pow_op);
    return mlir::success();
  }
};

struct SoftmaxOpLowering : mlir::OpConversionPattern<SoftmaxOp> {
  using mlir::OpConversionPattern<SoftmaxOp>::OpConversionPattern;

  auto matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto result_tensor_type = op.getResult().getType();

    auto init_op =
        mlir::tensor::EmptyOp::create(rewriter, loc, result_tensor_type, {});

    auto softmax_op = mlir::linalg::SoftmaxOp::create(
        rewriter, loc, mlir::TypeRange{result_tensor_type},
        adaptor.getOperand(), init_op, op.getDim());

    rewriter.replaceOp(op, softmax_op);
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
      return mlir::TupleType::get(ctx, {memref_type});
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
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::BuiltinDialect, mlir::arith::ArithDialect,
                    mlir::tensor::TensorDialect, mlir::math::MathDialect>();
  }

  auto runOnOperation() -> void final {
    auto& context = getContext();

    mlir::ConversionTarget target(context);

    target.addLegalDialect<
        mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
        mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
        mlir::bufferization::BufferizationDialect, mlir::BuiltinDialect,
        mlir::tensor::TensorDialect, mlir::math::MathDialect>();

    target.addLegalOp<TupleAccessOp>();
    target.addIllegalDialect<AxonDialect>();

    AxonToStandardTypeConverter type_converter{&context};

    mlir::RewritePatternSet patterns{&context};

    // clang-format off
    patterns.add<
      GetDataOpLowering, 
      GetGradOpLowering,
      AccumulateOpLowering, 

      AddOpLowering, 
      MulOpLowering, 
      SubOpLowering,
      MatMulOpLowering, 
      NegOpLowering,
      PowOpLowering,
      SoftmaxOpLowering,

      TransposeOpLowering, 
      SumOpLowering,
      ScalarMulOpLowering,

      ConstantOpLowering,
      FillOpLowering,
      ExpandDimsOpLowering, 
      SqueezeOpLowering,
      UnqueezeOpLowering, 
      ReshapeOpLowering
    >(type_converter, &context);
    // clang-format on

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
