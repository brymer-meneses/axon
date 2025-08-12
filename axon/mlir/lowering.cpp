module;

#include "axon/mlir/dialect/dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
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
    auto rhs_tensor = llvm::cast<mlir::TensorType>(adaptor.getRhs().getType());

    if (rhs_tensor.getShape().equals(lhs_tensor.getShape())) {
      auto new_op = rewriter.create<LoweredBinaryOp>(loc, adaptor.getLhs(),
                                                     adaptor.getRhs());
      rewriter.replaceOp(op, new_op);
      return mlir::success();
    }

    std::unreachable();
  }
};

struct GetDataLowering : mlir::OpConversionPattern<GetDataOp> {
  using mlir::OpConversionPattern<GetDataOp>::OpConversionPattern;

  auto matchAndRewrite(GetDataOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();
    auto tensor_ref = llvm::cast<TensorRefType>(op.getInput().getType());

    if (not tensor_ref.getRequiresGrad()) {
      auto new_op = rewriter.create<mlir::bufferization::ToTensorOp>(
          loc, adaptor.getInput());
      rewriter.replaceOp(op, new_op);
      return mlir::success();
    }

    auto memref = rewriter.create<TupleAccessOp>(loc, adaptor.getInput(), 0);
    auto new_op = rewriter.create<mlir::bufferization::ToTensorOp>(loc, memref);
    rewriter.replaceOp(op, new_op);
    return mlir::success();
  }
};

struct AccumulateGradLowering : mlir::OpConversionPattern<AccumulateGradOp> {
  using mlir::OpConversionPattern<AccumulateGradOp>::OpConversionPattern;

  auto matchAndRewrite(AccumulateGradOp op, OpAdaptor adaptor,
                       mlir::ConversionPatternRewriter& rewriter) const
      -> mlir::LogicalResult final {
    auto loc = op.getLoc();

    auto tensor_type =
        llvm::dyn_cast<TensorRefType>(op.getAccumulator().getType());
    // We know that `adaptor.getAccumulator` must require a gradient. Since
    // `TensorRefType` are lowered to tuple<memref, memref> then we need to
    // access the second element.
    auto grad_ref =
        rewriter.create<TupleAccessOp>(loc, adaptor.getAccumulator(), 1);
    auto memref_type = mlir::MemRefType::get(tensor_type.getShape(),
                                             tensor_type.getElementType());
    auto value_as_memref = rewriter.create<mlir::bufferization::ToMemrefOp>(
        loc, memref_type, adaptor.getValue());

    rewriter.create<mlir::linalg::AddOp>(
        loc, mlir::ValueRange{grad_ref, value_as_memref},
        mlir::ValueRange{grad_ref});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct AxonToStdTypeConverter : mlir::TypeConverter {
  AxonToStdTypeConverter(mlir::MLIRContext* ctx) {
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
    return "axon-lowering";
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
    target.addIllegalOp<AddOp, MulOp, GetDataOp, AccumulateGradOp>();

    AxonToStdTypeConverter type_converter{&context};

    mlir::RewritePatternSet patterns{&context};
    patterns.add<AddOpLowering, MulOpLowering, GetDataLowering,
                 AccumulateGradLowering>(type_converter, &context);

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

export auto createAxonToStandardLoweringPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<AxonToStandardLoweringPass>();
}

}  // namespace axon
