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

struct AccumulateOpLowering : public mlir::OpConversionPattern<AccumulateOp> {
  using OpConversionPattern<AccumulateOp>::OpConversionPattern;

  auto matchAndRewrite(AccumulateOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto tensor = llvm::dyn_cast<mlir::TensorType>(op.getRhs().getType());
    if (!tensor) {
      return mlir::failure();
    }

    auto memref_type =
        mlir::MemRefType::get(tensor.getShape(), tensor.getElementType());

    auto tensor_memref = rewriter.create<mlir::bufferization::ToMemrefOp>(
        op.getLoc(), memref_type, op.getRhs());

    rewriter.create<mlir::linalg::AddOp>(
        op.getLoc(), mlir::ValueRange{op.getLhs(), tensor_memref},
        mlir::ValueRange{op.getLhs()});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct AxonLoweringPass
    : public mlir::PassWrapper<AxonLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
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
    target.addIllegalOp<AccumulateOp>();

    mlir::RewritePatternSet patterns{&context};

    patterns.add<AccumulateOpLowering>(&context);

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
