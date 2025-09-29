#include <utility>

#include "axon/base/macros.h"
#include "axon/mlir/dialect/dialect.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace axon {

namespace {

// (x^T)^T => x
struct EliminateRedundantTransposePattern
    : mlir::OpRewritePattern<TransposeOp> {
  using mlir::OpRewritePattern<TransposeOp>::OpRewritePattern;

  auto matchAndRewrite(TransposeOp inner, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto outer = inner.getInput().getDefiningOp<TransposeOp>();
    if (!outer) {
      return mlir::failure();
    }

    if (inner.getFrom() == outer.getTo() && inner.getTo() == outer.getFrom()) {
      rewriter.replaceOp(inner, outer.getInput());
      return mlir::success();
    }

    return mlir::failure();
  }
};

/// x - x => 0
struct EliminateSelfSubtractionPattern : mlir::OpRewritePattern<SubOp> {
  using mlir::OpRewritePattern<SubOp>::OpRewritePattern;

  auto matchAndRewrite(SubOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    if (op.getLhs() != op.getRhs()) {
      return mlir::failure();
    }

    auto loc = op.getLoc();
    auto fill_type = op.getLhs().getType();
    auto element_type =
        mlir::cast<mlir::RankedTensorType>(fill_type).getElementType();
    auto zero_attr = mlir::FloatAttr::get(element_type, 0.0);
    auto zero = FillOp::create(rewriter, loc, fill_type, zero_attr);

    rewriter.replaceOp(op, zero);
    return mlir::success();
  }
};

/// x + (-x) => 0
/// (-x) + x => 0
struct EliminateAdditionOfSelfNegative : mlir::OpRewritePattern<AddOp> {
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

  auto matchAndRewrite(AddOp op, mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    if (auto lhs_neg = op.getLhs().getDefiningOp<NegOp>()) {
      if (lhs_neg.getInput() == op.getRhs()) {
        auto fill_type = op.getResult().getType();
        auto element_type =
            mlir::cast<mlir::RankedTensorType>(fill_type).getElementType();
        auto zero_attr = mlir::FloatAttr::get(element_type, 0.0);
        auto zero = FillOp::create(rewriter, op.getLoc(), fill_type, zero_attr);
        rewriter.replaceOp(op, zero);
        return mlir::success();
      }
    }

    if (auto rhs_neg = op.getRhs().getDefiningOp<NegOp>()) {
      if (rhs_neg.getInput() == op.getLhs()) {
        auto fill_type = op.getResult().getType();
        auto element_type =
            mlir::cast<mlir::RankedTensorType>(fill_type).getElementType();
        auto zero_attr = mlir::FloatAttr::get(element_type, 0.0);
        auto zero = FillOp::create(rewriter, op.getLoc(), fill_type, zero_attr);
        rewriter.replaceOp(op, zero);
        return mlir::success();
      }
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
      lhs = transpose_lhs.getInput();
    }
    if (auto transpose_rhs = op.getRhs().getDefiningOp<TransposeOp>()) {
      should_transpose_right = true;
      rhs = transpose_rhs.getInput();
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
        op = expand_dims_op.getInput();
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

auto AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                        mlir::MLIRContext* context) -> void {
  patterns.add<EliminateAdditionOfSelfNegative>(context);
}

auto TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                              mlir::MLIRContext* context)
    -> void {
  patterns.add<EliminateRedundantTransposePattern>(context);
}

auto MatMulOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                           mlir::MLIRContext* context) -> void {
  patterns.add<FuseTransposePattern>(context);
}

auto SubOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                        mlir::MLIRContext* context) -> void {
  patterns.add<EliminateSelfSubtractionPattern>(context);
}

auto AccumulateOp::canonicalize(AccumulateOp op,
                                mlir::PatternRewriter& rewriter)
    -> mlir::LogicalResult {
  auto element_type = op.getSource().getType().getElementType();
  auto constant_op = op.getSource().getDefiningOp<ConstantOp>();
  if (!constant_op) {
    return mlir::failure();
  }

  auto value = constant_op.getValue();
  if (!value.isSplat()) {
    return mlir::failure();
  }

  if (element_type.isFloat()) {
    auto float_value = value.getSplatValue<llvm::APFloat>();
    if (float_value.isZero()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
  }

  return mlir::failure();
}

using PoisonAttr = void;

template <typename Op, typename Callback>
static auto handleElementWiseBinaryFold(typename Op::FoldAdaptor adaptor,
                                        mlir::Type element_type,
                                        Callback callback)
    -> mlir::OpFoldResult {
  auto lhs =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getLhs());
  auto rhs =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getRhs());
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (element_type.isFloat()) {
    return constFoldBinaryOp<mlir::FloatAttr, mlir::APFloat, PoisonAttr>(
        adaptor.getOperands(), callback);
  } else if (element_type.isInteger()) {
    return constFoldBinaryOp<mlir::IntegerAttr, mlir::APInt, PoisonAttr>(
        adaptor.getOperands(), callback);
  }

  return nullptr;
}

static auto isSplatWithValue(mlir::DenseElementsAttr attr, double value) {
  auto element_type = attr.getElementType();
  if (!attr.isSplat()) {
    return false;
  }

  if (element_type.isInteger()) {
    return attr.getSplatValue<llvm::APInt>() == static_cast<int64_t>(value);
  } else if (element_type.isFloat()) {
    return attr.getSplatValue<llvm::APFloat>().isExactlyValue(value);
  }
  return false;
}

auto FillOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
  return mlir::DenseElementsAttr::get(getType(), adaptor.getFillValueAttr());
}

auto ConstantOp::fold(FoldAdaptor) -> mlir::OpFoldResult { return getValue(); }

auto AddOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
  auto fill_type = getResult().getType();
  if (auto lhs_neg = getLhs().getDefiningOp<NegOp>()) {
    if (lhs_neg.getInput() == getRhs()) {
      return mlir::DenseElementsAttr::get(fill_type, 0.0);
    }
  }
  if (auto rhs_neg = getRhs().getDefiningOp<NegOp>()) {
    if (rhs_neg.getInput() == getLhs()) {
      return mlir::DenseElementsAttr::get(fill_type, 0.0);
    }
  }

  return handleElementWiseBinaryFold<AddOp>(
      adaptor, getType().getElementType(),
      [](auto lhs, auto rhs) { return lhs + rhs; });
}

auto MulOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
  auto lhs =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getLhs());
  auto rhs =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getRhs());
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (isSplatWithValue(lhs, 0) || isSplatWithValue(rhs, 0)) {
    return mlir::DenseElementsAttr::get(getResult().getType(), 0.0);
  }

  if (isSplatWithValue(lhs, 1)) {
    return getRhs();
  }
  if (isSplatWithValue(rhs, 1)) {
    return getLhs();
  }

  return handleElementWiseBinaryFold<MulOp>(
      adaptor, getType().getElementType(),
      [](auto lhs, auto rhs) { return lhs * rhs; });
}

auto MatMulOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
  auto lhs =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getLhs());
  auto rhs =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getRhs());
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (isSplatWithValue(lhs, 0) || isSplatWithValue(rhs, 0)) {
    return mlir::DenseElementsAttr::get(getResult().getType(), 0.0);
  }

  return nullptr;
}

auto SubOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
  auto lhs =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getLhs());
  auto rhs =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getRhs());
  if (!lhs || !rhs) {
    return nullptr;
  }

  return handleElementWiseBinaryFold<SubOp>(
      adaptor, getType().getElementType(),
      [](auto lhs, auto rhs) { return lhs - rhs; });
}

auto ScalarMulOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
  auto elements =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getInput());
  if (!elements) {
    return nullptr;
  }

  auto scalar = adaptor.getScalar();

  if (auto float_attr = mlir::dyn_cast_if_present<mlir::FloatAttr>(scalar)) {
    auto value = float_attr.getValue();
    auto elems = elements.getValues<llvm::APFloat>()[0];

    AXON_DCHECK(&value.getSemantics() == &elems.getSemantics(),
                "ScalarMulOp::fold: mismatched APFloat semantics");
    return constFoldUnaryOp<mlir::FloatAttr, mlir::APFloat, PoisonAttr>(
        adaptor.getOperands(),
        [value](const mlir::APFloat& elem) -> mlir::APFloat {
          return elem * value;
        });
  }
  return nullptr;
};

auto ReshapeOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
  auto elements =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getInput());
  if (!elements) {
    return nullptr;
  }
  auto shape_type = mlir::RankedTensorType::get(adaptor.getTargetShape(),
                                                elements.getElementType());
  return elements.reshape(shape_type);
}

auto NegOp::fold(FoldAdaptor adaptor) -> mlir::OpFoldResult {
  auto elements =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(adaptor.getInput());
  if (!elements) {
    return nullptr;
  }

  return constFoldUnaryOp<mlir::FloatAttr, mlir::APFloat, PoisonAttr>(
      adaptor.getInput(),
      [](const mlir::APFloat& elem) -> mlir::APFloat { return -elem; });
};

}  // namespace axon
