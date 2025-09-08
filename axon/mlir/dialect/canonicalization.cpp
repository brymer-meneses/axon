#include "axon/mlir/dialect/dialect.h"
#include "mlir/Support/LLVM.h"

namespace axon {

namespace {

struct EliminateIdentityMulPattern : mlir::OpRewritePattern<MulOp> {
  using mlir::OpRewritePattern<MulOp>::OpRewritePattern;
  auto matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto lhs = op.getLhs().getDefiningOp<FillOp>();
    if (!lhs) {
      return mlir::failure();
    }

    if (lhs.getFillValue().isExactlyValue(1.0)) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }

    auto rhs = op.getLhs().getDefiningOp<FillOp>();
    if (!rhs) {
      return mlir::failure();
    }
    if (rhs.getFillValue().isExactlyValue(1.0)) {
      rewriter.replaceOp(op, op.getLhs());
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct EliminateIdentityAddPattern : mlir::OpRewritePattern<AddOp> {
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

  auto matchAndRewrite(AddOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto lhs = op.getLhs().getDefiningOp<FillOp>();
    if (!lhs) {
      return mlir::failure();
    }

    if (lhs.getFillValue().isExactlyValue(0.0)) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }

    auto rhs = op.getLhs().getDefiningOp<FillOp>();
    if (!rhs) {
      return mlir::failure();
    }
    if (rhs.getFillValue().isExactlyValue(0.0)) {
      rewriter.replaceOp(op, op.getLhs());
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct EliminateRedundantTransposePattern
    : mlir::OpRewritePattern<TransposeOp> {
  using mlir::OpRewritePattern<TransposeOp>::OpRewritePattern;

  auto matchAndRewrite(TransposeOp inner, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto outer = inner.getOperand().getDefiningOp<TransposeOp>();
    if (!outer) {
      return mlir::failure();
    }

    if (inner.getFrom() == outer.getTo() && inner.getTo() == outer.getFrom()) {
      rewriter.replaceOp(inner, outer.getOperand());
      return mlir::success();
    }

    return mlir::failure();
  }
};

/// x - x => x
struct EliminateSelfSubtractionPattern : mlir::OpRewritePattern<SubOp> {
  using mlir::OpRewritePattern<SubOp>::OpRewritePattern;

  auto matchAndRewrite(SubOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    if (op.getLhs() != op.getRhs()) {
      return mlir::failure();
    }

    auto loc = op.getLoc();
    auto fill_type = op.getLhs().getType();
    auto zero =
        FillOp::create(rewriter, loc, fill_type, rewriter.getF32FloatAttr(0.0));

    rewriter.replaceOp(op, zero);
    return mlir::success();
  }
};

struct SimplifyZeroSubtractionPattern : mlir::OpRewritePattern<SubOp> {
  using mlir::OpRewritePattern<SubOp>::OpRewritePattern;

  auto matchAndRewrite(SubOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto lhs = op.getLhs().getDefiningOp<FillOp>();
    if (!lhs) {
      return mlir::failure();
    }
    if (!lhs.getFillValue().isExactlyValue(0.0)) {
      return mlir::failure();
    }

    auto neg_op = NegOp::create(rewriter, op.getLoc(), op.getRhs());
    rewriter.replaceOp(op, neg_op);
    return mlir::success();
  }
};

struct SimplifyMultiplicationOfFillOpPattern : mlir::OpRewritePattern<MulOp> {
  using mlir::OpRewritePattern<MulOp>::OpRewritePattern;

  auto matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto lhs = op.getLhs().getDefiningOp<FillOp>();
    auto rhs = op.getRhs().getDefiningOp<FillOp>();
    if (!lhs || !rhs) {
      return mlir::failure();
    }

    // FIXME: These conversions can overflow.
    auto lhs_value = lhs.getFillValue().convertToDouble();
    auto rhs_value = lhs.getFillValue().convertToDouble();

    auto loc = op.getLoc();
    auto prod = lhs_value * rhs_value;
    auto fill_op = FillOp::create(rewriter, loc, op.getResult().getType(),
                                  rewriter.getF64FloatAttr(prod));

    rewriter.replaceOp(op, fill_op);
    return mlir::success();
  }
};

struct SimplifyNegationOfFillOpPattern : mlir::OpRewritePattern<NegOp> {
  using mlir::OpRewritePattern<NegOp>::OpRewritePattern;

  auto matchAndRewrite(NegOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto operand = op.getOperand().getDefiningOp<FillOp>();
    if (!operand) {
      return mlir::failure();
    }

    llvm::APFloat scalar = operand.getFillValue();
    scalar.changeSign();

    auto fill_op =
        FillOp::create(rewriter, op.getLoc(), op.getResult().getType(), scalar);
    rewriter.replaceOp(op, fill_op);
    return mlir::success();
  }
};

struct SimplifyScalarMultiplicationPattern
    : mlir::OpRewritePattern<ScalarMulOp> {
  using mlir::OpRewritePattern<ScalarMulOp>::OpRewritePattern;

  auto matchAndRewrite(ScalarMulOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto fill_op = op.getOperand().getDefiningOp<FillOp>();
    if (!fill_op) {
      return mlir::failure();
    }

    auto loc = op.getLoc();
    if (fill_op.getFillValue().isExactlyValue(-1)) {
      auto neg_op = NegOp::create(rewriter, loc, op.getOperand());
      rewriter.replaceOp(op, neg_op);
      return mlir::success();
    }

    llvm::APFloat scalar = op.getScalar();
    llvm::APFloat fill_value = fill_op.getFillValue();
    auto status =
        scalar.multiply(fill_value, llvm::RoundingMode::NearestTiesToEven);
    if (status != llvm::detail::opStatus::opOK) {
      return mlir::failure();
    }

    // we use `scalar` here since the multiply result is stored on the lhs.
    auto new_fill_op =
        FillOp::create(rewriter, loc, op.getResult().getType(), scalar);
    rewriter.replaceOp(op, new_fill_op);
    return mlir::success();
  }
};

struct FuseTransposePattern : mlir::OpRewritePattern<MatMulOp> {
  using mlir::OpRewritePattern<MatMulOp>::OpRewritePattern;

  auto matchAndRewrite(MatMulOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto should_transpose_left = op.getTransposeLhs();
    auto should_transpose_right = op.getTransposeRhs();

    if (auto transpose_lhs = op.getLhs().getDefiningOp<TransposeOp>()) {
      should_transpose_left = true;
      lhs = transpose_lhs.getOperand();
    }
    if (auto transpose_rhs = op.getRhs().getDefiningOp<TransposeOp>()) {
      should_transpose_right = true;
      rhs = transpose_rhs.getOperand();
    }

    if (lhs == op.getLhs() && rhs == op.getRhs()) {
      return mlir::failure();
    }

    auto new_op =
        MatMulOp::create(rewriter, op.getLoc(), op.getResult().getType(), lhs,
                         rhs, should_transpose_left, should_transpose_right);
    rewriter.replaceOp(op, new_op);
    return mlir::success();
  }
};

struct FuseExpandedDimsPattern : mlir::OpRewritePattern<MatMulOp> {
  using mlir::OpRewritePattern<MatMulOp>::OpRewritePattern;

  auto matchAndRewrite(MatMulOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto get_expand_dims_info = [](mlir::Value op)
        -> std::pair<mlir::Value, llvm::SmallVector<int64_t>> {
      llvm::SmallVector<int64_t> dims;
      if (auto expand_dims_op = op.getDefiningOp<ExpandDimsOp>()) {
        for (auto mapping : expand_dims_op.getMappings()) {
          auto pair = mlir::cast<mlir::ArrayAttr>(mapping);
          auto dim = mlir::cast<mlir::IntegerAttr>(pair[0]).getInt();
          dims.push_back(dim);
        }
        op = expand_dims_op.getOperand();
      }
      return {op, dims};
    };

    auto [lhs_op, lhs_expanded_dims] = get_expand_dims_info(op.getLhs());
    auto [rhs_op, rhs_expanded_dims] = get_expand_dims_info(op.getRhs());

    if (lhs_expanded_dims.empty() && rhs_expanded_dims.empty()) {
      return mlir::failure();
    }

    bool should_transpose_lhs = op.getTransposeLhs();
    bool should_transpose_rhs = op.getTransposeRhs();
    auto result_type = op.getResult().getType();

    auto new_matmul_op = MatMulOp::create(
        rewriter, loc, result_type, lhs_op, rhs_op, should_transpose_lhs,
        should_transpose_rhs, lhs_expanded_dims, rhs_expanded_dims);

    rewriter.replaceOp(op, new_matmul_op);
    return mlir::success();
  }
};

}  // namespace

auto MulOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                        mlir::MLIRContext* context) -> void {
  patterns
      .add<EliminateIdentityMulPattern, SimplifyMultiplicationOfFillOpPattern>(
          context);
}

auto AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                        mlir::MLIRContext* context) -> void {
  patterns.add<EliminateIdentityAddPattern>(context);
}

auto TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                              mlir::MLIRContext* context)
    -> void {
  patterns.add<EliminateRedundantTransposePattern>(context);
}

auto MatMulOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                           mlir::MLIRContext* context) -> void {
  patterns.add<FuseTransposePattern, FuseExpandedDimsPattern>(context);
}

auto SubOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                        mlir::MLIRContext* context) -> void {
  patterns.add<EliminateSelfSubtractionPattern, SimplifyZeroSubtractionPattern>(
      context);
}

auto NegOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                        mlir::MLIRContext* context) -> void {
  patterns.add<SimplifyNegationOfFillOpPattern>(context);
}

auto ScalarMulOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                              mlir::MLIRContext* context)
    -> void {
  patterns.add<SimplifyScalarMultiplicationPattern>(context);
}

}  // namespace axon
