module;

#include "axon/mlir/dialect/dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

export module axon.mlir:lowering;

import std;

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
    auto rhs_tensor = llvm::cast<mlir::TensorType>(adaptor.getLhs().getType());

    if (rhs_tensor.getShape().equals(lhs_tensor.getShape())) {
      auto new_op = rewriter.create<LoweredBinaryOp>(loc, adaptor.getLhs(),
                                                     adaptor.getRhs());
      rewriter.replaceOp(op, new_op);
      return mlir::success();
    }

    std::unreachable();
  }
};

using AddOpLowering = BinaryOpLowering<AddOp, mlir::arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<MulOp, mlir::arith::MulFOp>;

struct AxonLoweringPass
    : mlir::PassWrapper<AxonLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AxonLoweringPass)

  auto getArgument() const -> llvm::StringRef override {
    return "axon-lowering";
  }

  auto getDependentDialects(mlir::DialectRegistry& registry) const
      -> void override {
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::arith::ArithDialect, mlir::tensor::TensorDialect>();
  }

  auto runOnOperation() -> void final {
    auto& context = getContext();
    mlir::ConversionTarget target(context);

    target
        .addLegalDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                         mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                         mlir::bufferization::BufferizationDialect,
                         mlir::tensor::TensorDialect>();
    target.addIllegalOp<AddOp, MulOp>();

    mlir::RewritePatternSet patterns{&context};
    patterns.add<AddOpLowering, MulOpLowering>(&context);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

export auto createAxonLoweringPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<AxonLoweringPass>();
}

}  // namespace axon
