#include "axon/mlir/dialect/dialect.h"
#include "mlir/Support/LLVM.h"

namespace axon {

namespace {

struct EliminateIdentityMulPattern : mlir::OpRewritePattern<MulOp> {
  using mlir::OpRewritePattern<MulOp>::OpRewritePattern;
  auto matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto lhs = op.getLhs().getDefiningOp<FillLikeOp>();
    if (!lhs) {
      return mlir::failure();
    }

    if (lhs.getFillValue().isExactlyValue(1.0)) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }

    auto rhs = op.getLhs().getDefiningOp<FillLikeOp>();
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
    auto lhs = op.getLhs().getDefiningOp<FillLikeOp>();
    if (!lhs) {
      return mlir::failure();
    }

    if (lhs.getFillValue().isExactlyValue(0.0)) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }

    auto rhs = op.getLhs().getDefiningOp<FillLikeOp>();
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

struct FuseBroadcastPattern : mlir::OpRewritePattern<MatMulOp> {
  using mlir::OpRewritePattern<MatMulOp>::OpRewritePattern;

  auto matchAndRewrite(MatMulOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto get_broadcast_info = [](mlir::Value op)
        -> std::pair<mlir::Value, llvm::SmallVector<int64_t>> {
      llvm::SmallVector<int64_t> dims;
      if (auto broadcast_op = op.getDefiningOp<BroadcastOp>()) {
        for (auto expansion : broadcast_op.getExpansions()) {
          auto pair = mlir::cast<mlir::ArrayAttr>(expansion);
          auto dim = mlir::cast<mlir::IntegerAttr>(pair[0]).getInt();
          dims.push_back(dim);
        }
        op = broadcast_op.getOperand();
      }
      return {op, dims};
    };

    auto [lhs_op, lhs_broadcasted_dims] = get_broadcast_info(op.getLhs());
    auto [rhs_op, rhs_broadcasted_dims] = get_broadcast_info(op.getRhs());

    if (lhs_broadcasted_dims.empty() && rhs_broadcasted_dims.empty()) {
      return mlir::failure();
    }

    bool should_transpose_lhs = op.getTransposeLhs();
    bool should_transpose_rhs = op.getTransposeRhs();
    auto result_type = op.getResult().getType();

    auto new_matmul_op = MatMulOp::create(
        rewriter, loc, result_type, lhs_op, rhs_op, should_transpose_lhs,
        should_transpose_rhs, lhs_broadcasted_dims, rhs_broadcasted_dims);

    rewriter.replaceOp(op, new_matmul_op);
    return mlir::success();
  }
};

}  // namespace

auto MulOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                        mlir::MLIRContext* context) -> void {
  patterns.add<EliminateIdentityMulPattern>(context);
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
  patterns.add<FuseTransposePattern, FuseBroadcastPattern>(context);
}

}  // namespace axon
