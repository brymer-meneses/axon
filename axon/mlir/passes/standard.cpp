module;

#include <algorithm>
#include <limits>
#include <optional>
#include <ranges>
#include <utility>

#include "axon/base/macros.h"
#include "axon/mlir/dialect/dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

export module axon.mlir.passes:standard_lowering;

import axon.base;

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
    auto empty_op =
        mlir::tensor::EmptyOp::create(rewriter, loc, tensor_type, {});

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

    llvm::SmallVector<i64> dimensions_to_expand;
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
        /*inputs=*/mlir::ValueRange{adaptor.getInput()},
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

struct SqueezeOpLowering : mlir::OpConversionPattern<SqueezeOp> {
  using mlir::OpConversionPattern<SqueezeOp>::OpConversionPattern;

  auto matchAndRewrite(SqueezeOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto input_tensor =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
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
        rewriter, op.getLoc(), adaptor.getInput(), *reassociation_indices);
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
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
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
        rewriter, op.getLoc(), op.getResult().getType(), adaptor.getInput(),
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
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
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
          rewriter, op.getLoc(), op.getResult().getType(), adaptor.getInput(),
          *reassociation_indices);
      rewriter.replaceOp(op, expand_op);
      return mlir::success();
    }

    if (input_tensor.getRank() > result_tensor.getRank()) {
      auto expand_op = mlir::tensor::CollapseShapeOp::create(
          rewriter, op.getLoc(), op.getResult().getType(), adaptor.getInput(),
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

    auto init_op = adaptor.getInput();

    auto transpose_op = mlir::linalg::GenericOp::create(
        rewriter, op.getLoc(), mlir::TypeRange{result_type},
        mlir::ValueRange{adaptor.getInput()}, mlir::ValueRange{init_op},
        indexing_maps, iterator_types,
        [](mlir::OpBuilder& builder, mlir::Location loc,
           mlir::ValueRange args) {
          mlir::linalg::YieldOp::create(builder, loc, args[0]);
        });

    rewriter.replaceOp(op, transpose_op.getResult(0));
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

    auto sink_memref_type = mlir::MemRefType::get(
        sink_tensor_type.getShape(), sink_tensor_type.getElementType());
    auto sink_as_memref = mlir::bufferization::ToBufferOp::create(
        rewriter, loc, sink_memref_type, sink);

    mlir::bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, std::nullopt, accumulation_op, sink_as_memref,
        /*restrict=*/false, /*writable=*/true);

    rewriter.eraseOp(op);
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
    auto tensor = mlir::cast<mlir::RankedTensorType>(op.getInput().getType());

    auto init_op = createZerosLike(rewriter, tensor, loc);

    auto neg_op = mlir::linalg::NegFOp::create(
        rewriter, loc, mlir::ValueRange{adaptor.getInput()},
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
    auto result_type = op.getInput().getType();

    auto build_mul = [&](mlir::TypedAttr scalar_attr) {
      auto scalar_constant =
          mlir::arith::ConstantOp::create(rewriter, loc, scalar_attr)
              .getResult();
      auto scalar_tensor = mlir::tensor::SplatOp::create(
          rewriter, loc, scalar_constant, result_type);

      auto empty_op =
          mlir::tensor::EmptyOp::create(rewriter, loc, result_type, {});
      auto prod = mlir::linalg::MulOp::create(
          rewriter, loc, mlir::ValueRange{scalar_tensor, adaptor.getInput()},
          mlir::ValueRange{empty_op});

      rewriter.replaceOp(op, prod.getResult(0));
      return mlir::success();
    };

    if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(op.getScalar())) {
      return build_mul(float_attr);
    }

    if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(op.getScalar())) {
      return build_mul(int_attr);
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

    return rewriter.notifyMatchFailure(op, "Unsupported element type");
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
        rewriter.getF32FloatAttr(op.getExponent().convertToFloat()));

    auto exponent_tensor = mlir::tensor::SplatOp::create(
        rewriter, loc, exponent_constant, result_tensor_type.getShape());

    auto pow_op = mlir::math::PowFOp::create(rewriter, loc, adaptor.getInput(),
                                             exponent_tensor);

    rewriter.replaceOp(op, pow_op);
    return mlir::success();
  }
};

static auto getReducedIdentityMap(i64 rank, llvm::ArrayRef<i64> excluded,
                                  bool keep_dims, mlir::MLIRContext* context)
    -> mlir::AffineMap {
  llvm::SmallVector<mlir::AffineExpr> affine_exprs;
  for (auto i : std::views::iota(0, rank)) {
    if (std::ranges::contains(excluded, i)) {
      // If we have keep_dims=True, then we should accumulate to the 0th
      // dimension in the output map. Otherwise, we don't have to push it back
      // to `affine_exprs`.
      if (keep_dims) {
        affine_exprs.push_back(mlir::getAffineConstantExpr(0, context));
      }
    } else {
      affine_exprs.push_back(mlir::getAffineDimExpr(i, context));
    }
  }

  return mlir::AffineMap::get(rank, /*symbolCount=*/0, affine_exprs, context);
}

template <typename ReduceOp>
static auto reduce(mlir::OpBuilder& rewriter, mlir::Location loc,
                   mlir::Value input, bool keep_dims, i64 axis,
                   mlir::arith::ConstantOp init_value) -> mlir::Value {
  AXON_DCHECK(mlir::isa<mlir::RankedTensorType>(input.getType()));

  auto tensor_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto rank = tensor_type.getRank();
  auto context = rewriter.getContext();

  llvm::SmallVector iterator_types(rank, mlir::utils::IteratorType::parallel);
  iterator_types[axis] = mlir::utils::IteratorType::reduction;

  auto output_map = getReducedIdentityMap(rank, {axis}, keep_dims, context);
  auto input_map = mlir::AffineMap::getMultiDimIdentityMap(rank, context);

  llvm::SmallVector<mlir::AffineMap> indexing_maps = {input_map, output_map};

  llvm::SmallVector<i64> result_shape(tensor_type.getShape());
  if (keep_dims) {
    result_shape[axis] = 1;
  } else {
    result_shape.erase(result_shape.begin() + axis);
  }

  auto result_type =
      mlir::RankedTensorType::get(result_shape, tensor_type.getElementType());

  auto init_op =
      mlir::tensor::SplatOp::create(rewriter, loc, init_value, result_shape);

  auto reduce_op = mlir::linalg::GenericOp::create(
      rewriter, loc, mlir::TypeRange{result_type}, mlir::ValueRange{input},
      mlir::ValueRange{init_op},

      indexing_maps, iterator_types,
      [](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) {
        auto reduce_op = ReduceOp::create(builder, loc, args[0], args[1]);
        mlir::linalg::YieldOp::create(builder, loc, reduce_op.getResult());
      });

  return reduce_op.getResult(0);
}

struct SoftmaxOpLowering : mlir::OpConversionPattern<SoftmaxOp> {
  using mlir::OpConversionPattern<SoftmaxOp>::OpConversionPattern;

  auto matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto element_type = op.getResult().getType().getElementType();
    auto axis = op.getDim();

    AXON_DCHECK(element_type.isFloat());

    auto init_value = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(element_type, -1e30));
    auto zero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(element_type));

    auto max =
        reduce<mlir::arith::MaxNumFOp>(rewriter, loc, input,
                                       /*keep_dims=*/false, axis, init_value);
    auto exp_diff = computeExpDiff(rewriter, loc, input, max, axis);
    auto denominator =
        reduce<mlir::arith::AddFOp>(rewriter, loc, exp_diff, false, axis, zero);

    auto softmax = computeSoftmax(rewriter, loc, exp_diff, denominator, axis);
    rewriter.replaceOp(op, softmax);
    return mlir::success();
  }

  static auto computeExpDiff(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value input, mlir::Value max, i64 axis)
      -> mlir::Value {
    auto context = input.getContext();
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto rank = input_type.getRank();

    llvm::SmallVector iterator_types(rank, mlir::utils::IteratorType::parallel);
    llvm::SmallVector indexing_maps = {
        // lhs map
        mlir::AffineMap::getMultiDimIdentityMap(rank, context),
        // rhs map
        getReducedIdentityMap(rank, {axis}, /*keep_dims=*/false, context),
        // output map
        mlir::AffineMap::getMultiDimIdentityMap(rank, context),
    };

    auto init_op = mlir::tensor::EmptyOp::create(builder, loc, input_type, {});

    auto generic_op = mlir::linalg::GenericOp::create(
        builder, loc, mlir::TypeRange{input_type}, mlir::ValueRange{input, max},
        mlir::ValueRange{init_op}, indexing_maps, iterator_types,
        [](mlir::OpBuilder& builder, mlir::Location loc,
           mlir::ValueRange args) {
          auto diff =
              mlir::arith::SubFOp::create(builder, loc, args[0], args[1]);
          auto exp = mlir::math::ExpOp::create(builder, loc, diff).getResult();
          mlir::linalg::YieldOp::create(builder, loc, exp);
        });

    return generic_op.getResult(0);
  }

  static auto computeSoftmax(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value numerator, mlir::Value denominator,
                             i64 axis) -> mlir::Value {
    auto context = numerator.getContext();
    auto input_type = mlir::cast<mlir::RankedTensorType>(numerator.getType());
    auto rank = input_type.getRank();

    llvm::SmallVector iterator_types(rank, mlir::utils::IteratorType::parallel);
    llvm::SmallVector indexing_maps = {
        // lhs map
        mlir::AffineMap::getMultiDimIdentityMap(rank, context),
        // rhs map
        getReducedIdentityMap(rank, {axis}, /*keep_dims=*/false, context),
        // output map
        mlir::AffineMap::getMultiDimIdentityMap(rank, context),
    };

    auto init_op = mlir::tensor::EmptyOp::create(builder, loc, input_type, {});

    auto generic_op = mlir::linalg::GenericOp::create(
        builder, loc, mlir::TypeRange{input_type},
        mlir::ValueRange{numerator, denominator}, mlir::ValueRange{init_op},
        indexing_maps, iterator_types,
        [](mlir::OpBuilder& builder, mlir::Location loc,
           mlir::ValueRange args) {
          auto div = mlir::arith::DivFOp::create(builder, loc, args[0], args[1])
                         .getResult();
          mlir::linalg::YieldOp::create(builder, loc, div);
        });

    return generic_op.getResult(0);
  }
};

struct CompareOpLowering : mlir::OpConversionPattern<CompareOp> {
  using mlir::OpConversionPattern<CompareOp>::OpConversionPattern;

  auto matchAndRewrite(CompareOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto result_type = op.getResult().getType();
    auto element_type = result_type.getElementType();

    auto mask_type = mlir::RankedTensorType::get(result_type.getShape(),
                                                 rewriter.getI1Type());

    if (element_type.isFloat()) {
      auto predicate =
          getFloatPredicate(static_cast<ComparePredicate>(op.getPredicate()));
      auto comparison_op =
          mlir::arith::CmpFOp::create(rewriter, loc, mask_type, predicate,
                                      adaptor.getLhs(), adaptor.getRhs());

      auto cast_op = mlir::arith::UIToFPOp::create(rewriter, loc, result_type,
                                                   comparison_op);
      rewriter.replaceOp(op, cast_op);
      return mlir::success();
    }

    if (element_type.isInteger()) {
      auto predicate =
          getIntPredicate(static_cast<ComparePredicate>(op.getPredicate()));
      auto comparison_op =
          mlir::arith::CmpIOp::create(rewriter, loc, mask_type, predicate,
                                      adaptor.getLhs(), adaptor.getRhs());

      // For i1 element types, the comparison result already matches the
      // expected tensor type. Otherwise, zero-extend to the target integer
      // width so we encode the mask as 0/1 values in the original dtype.
      if (element_type.getIntOrFloatBitWidth() == 1) {
        rewriter.replaceOp(op, comparison_op);
      } else {
        auto extended = mlir::arith::ExtUIOp::create(rewriter, loc, result_type,
                                                     comparison_op);
        rewriter.replaceOp(op, extended);
      }
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "Unsupported element type");
  }

  static auto getFloatPredicate(ComparePredicate predicate)
      -> mlir::arith::CmpFPredicate {
    switch (predicate) {
      case ComparePredicate::le:
        return mlir::arith::CmpFPredicate::ULE;
      case ComparePredicate::lt:
        return mlir::arith::CmpFPredicate::ULT;
      case ComparePredicate::ge:
        return mlir::arith::CmpFPredicate::UGE;
      case ComparePredicate::gt:
        return mlir::arith::CmpFPredicate::UGT;
      case ComparePredicate::eq:
        return mlir::arith::CmpFPredicate::UEQ;
      case ComparePredicate::ne:
        return mlir::arith::CmpFPredicate::UNE;
    }

    AXON_UNREACHABLE("Unsupported float compare predicate");
  }

  static auto getIntPredicate(ComparePredicate predicate)
      -> mlir::arith::CmpIPredicate {
    switch (predicate) {
      case ComparePredicate::le:
        return mlir::arith::CmpIPredicate::sle;
      case ComparePredicate::lt:
        return mlir::arith::CmpIPredicate::slt;
      case ComparePredicate::ge:
        return mlir::arith::CmpIPredicate::sge;
      case ComparePredicate::gt:
        return mlir::arith::CmpIPredicate::sgt;
      case ComparePredicate::eq:
        return mlir::arith::CmpIPredicate::eq;
      case ComparePredicate::ne:
        return mlir::arith::CmpIPredicate::ne;
    }

    AXON_UNREACHABLE("Unsupported integer compare predicate");
  }
};

struct ReluOpLowering : mlir::OpConversionPattern<ReluOp> {
  using mlir::OpConversionPattern<ReluOp>::OpConversionPattern;

  auto matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto result_type = op.getResult().getType();
    auto rank = result_type.getRank();
    auto context = op.getContext();

    llvm::SmallVector indexing_maps = {
        mlir::AffineMap::getMultiDimIdentityMap(rank, context),
        mlir::AffineMap::getMultiDimIdentityMap(rank, context),
    };

    llvm::SmallVector iterator_types(rank, mlir::utils::IteratorType::parallel);

    auto init_op =
        mlir::tensor::EmptyOp::create(rewriter, loc, result_type, {});

    auto zero_constant =
        mlir::arith::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(result_type.getElementType()))
            .getResult();

    auto generic_op = mlir::linalg::GenericOp::create(
        rewriter, loc, mlir::TypeRange{result_type},
        mlir::ValueRange{adaptor.getInput()}, mlir::ValueRange{init_op},
        indexing_maps, iterator_types,
        [zero_constant](mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::ValueRange args) {
          auto result_type = args[0].getType();
          auto relu = mlir::arith::MaxNumFOp::create(
                          builder, loc, result_type, zero_constant, args[0],
                          mlir::arith::FastMathFlags::none)
                          .getResult();

          mlir::linalg::YieldOp::create(builder, loc, relu);
        });

    rewriter.replaceOp(op, generic_op.getResult(0));
    return mlir::success();
  }
};

struct ArgMaxOpLowering : mlir::OpConversionPattern<ArgMaxOp> {
  using mlir::OpConversionPattern<ArgMaxOp>::OpConversionPattern;

  auto matchAndRewrite(ArgMaxOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto result_type =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto input_type =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());

    auto input_element_type = input_type.getElementType();
    auto result_element_type = result_type.getElementType();

    if (!input_element_type.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Expected input_element_type to be either int or float");
    }

    auto rank = input_type.getRank();
    auto context = op.getContext();
    auto axis = static_cast<i64>(op.getDim());
    auto keep_dims = op.getKeepDims();

    auto init_op = createZerosLike(rewriter, result_type, loc);
    auto maximum_values = computeMaximumValues(
        rewriter, loc, adaptor.getInput(), input_element_type, axis);

    llvm::SmallVector indexing_maps = {
        mlir::AffineMap::getMultiDimIdentityMap(rank, context),
        // `maximum_values` is computed with `keep_dims=false`.
        getReducedIdentityMap(rank, {axis}, /*keep_dims=*/false, context),
        getReducedIdentityMap(rank, {axis}, keep_dims, context),
    };

    llvm::SmallVector iterator_types(rank, mlir::utils::IteratorType::parallel);
    iterator_types[axis] = mlir::utils::IteratorType::reduction;

    auto argmax_op = mlir::linalg::GenericOp::create(
        rewriter, loc, mlir::TypeRange{result_type},
        mlir::ValueRange{adaptor.getInput(), maximum_values},
        mlir::ValueRange{init_op}, indexing_maps, iterator_types,
        [input_element_type, result_element_type, axis](
            mlir::OpBuilder& builder, mlir::Location loc,
            mlir::ValueRange args) {
          auto in = args[0];
          auto max = args[1];

          auto this_index = mlir::linalg::IndexOp::create(
                                builder, loc, builder.getI64IntegerAttr(axis))
                                .getResult();
          auto this_index_value =
              mlir::arith::IndexCastOp::create(builder, loc,
                                               result_element_type, this_index)
                  .getResult();

          mlir::Value is_max;
          if (input_element_type.isFloat()) {
            is_max = mlir::arith::CmpFOp::create(
                builder, loc, mlir::arith::CmpFPredicate::OEQ, in, max);
          } else {
            is_max = mlir::arith::CmpIOp::create(
                builder, loc, mlir::arith::CmpIPredicate::eq, in, max);
          }

          auto stored_index = args[2];
          auto value = mlir::arith::SelectOp::create(
              builder, loc, is_max, this_index_value, stored_index);

          mlir::linalg::YieldOp::create(builder, loc, value.getResult());
        });

    rewriter.replaceOp(op, argmax_op.getResult(0));
    return mlir::success();
  }

  static auto computeMaximumValues(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value input, mlir::Type element_type,
                                   i64 axis) -> mlir::Value {
    if (element_type.isFloat()) {
      auto init_value = mlir::arith::ConstantOp::create(
          builder, loc,
          builder.getFloatAttr(element_type,
                               -std::numeric_limits<f64>::infinity()));
      return reduce<mlir::arith::MaxNumFOp>(builder, loc, input,
                                            /*keep_dims=*/false, axis,
                                            init_value);
    } else if (element_type.isInteger()) {
      auto init_value = mlir::arith::ConstantOp::create(
          builder, loc,
          builder.getIntegerAttr(element_type,
                                 -std::numeric_limits<i64>::infinity()));
      return reduce<mlir::arith::MaxSIOp>(builder, loc, input,
                                          /*keep_dims=*/false, axis,
                                          init_value);
    }

    AXON_UNREACHABLE("Unhandled element_type");
  }
};

struct SumOpLowering : mlir::OpConversionPattern<SumOp> {
  using mlir::OpConversionPattern<SumOp>::OpConversionPattern;

  auto matchAndRewrite(SumOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto element_type = op.getResult().getType().getElementType();

    auto init_value = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(element_type));

    mlir::Value sum_op;
    if (element_type.isFloat()) {
      sum_op = reduce<mlir::arith::AddFOp>(rewriter, loc, adaptor.getInput(),
                                           op.getKeepDims(), op.getDim(),
                                           init_value);
    } else if (element_type.isInteger()) {
      sum_op = reduce<mlir::arith::AddIOp>(rewriter, loc, adaptor.getInput(),
                                           op.getKeepDims(), op.getDim(),
                                           init_value);
    } else {
      return rewriter.notifyMatchFailure(
          op, "sum currently supports integer or floating element types only");
    }

    rewriter.replaceOp(op, sum_op);
    return mlir::success();
  }
};

struct MeanOpLowering : mlir::OpConversionPattern<MeanOp> {
  using mlir::OpConversionPattern<MeanOp>::OpConversionPattern;

  auto matchAndRewrite(MeanOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto input_type = op.getInput().getType();
    auto result_type = op.getResult().getType();

    auto element_type = input_type.getElementType();
    AXON_DCHECK(element_type.isFloat());

    auto init_value = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(element_type));

    auto sum_op =
        reduce<mlir::arith::AddFOp>(rewriter, loc, adaptor.getInput(),
                                    op.getKeepDims(), op.getDim(), init_value);
    auto num_elements_along_axis =
        static_cast<f32>(input_type.getShape()[op.getDim()]);

    auto divisor = 1.0f / num_elements_along_axis;

    auto scaled = ScalarMulOp::create(rewriter, loc, sum_op,
                                      rewriter.getF32FloatAttr(divisor));

    rewriter.replaceOp(op, scaled);
    return mlir::success();
  }
};

struct AxonToStandardTypeConverter : mlir::TypeConverter {
  AxonToStandardTypeConverter(mlir::MLIRContext* ctx) {
    addConversion([](mlir::Type type) -> mlir::Type { return type; });
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

    target.addIllegalDialect<AxonDialect>();

    AxonToStandardTypeConverter type_converter{&context};

    mlir::RewritePatternSet patterns{&context};

    // clang-format off
    patterns.add<
      AccumulateOpLowering, 
      ArgMaxOpLowering,
      ReluOpLowering,
      CompareOpLowering,
      AddOpLowering, 
      MulOpLowering, 
      SubOpLowering,
      MeanOpLowering,
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
