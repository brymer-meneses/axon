module;

#include <algorithm>
#include <format>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "nanobind/nanobind.h"

export module axon.python:tensor_ops;

import axon.base;
import axon.core;

import :tensor;
import :jit;
import :tensor_impl;

namespace nb = nanobind;

namespace axon {

struct BroadcastInfo {
  llvm::SmallVector<insts::ExpandDims::Mapping> expand_dim_mappings;
  llvm::SmallVector<i64> unsqueezed_shape;
};

static auto tryGetBroadcastInfo(llvm::ArrayRef<i64> source_shape,
                                llvm::ArrayRef<i64> target_shape)
    -> std::optional<BroadcastInfo> {
  auto source_rank = static_cast<i64>(source_shape.size());
  auto target_rank = static_cast<i64>(target_shape.size());

  AXON_DCHECK(
      source_rank <= target_rank,
      "Expected the source shape to have lower rank than the target shape");

  BroadcastInfo broadcast_info;

  for (auto i = target_rank - 1; i >= 0; i -= 1) {
    // Calculate corresponding source index (right-aligned)
    auto source_index = i - (target_rank - source_rank);
    auto source_dim = source_index >= 0 ? source_shape[source_index] : 1;
    auto target_dim = target_shape[i];

    broadcast_info.unsqueezed_shape.push_back(source_dim);

    if (source_dim == 1) {
      broadcast_info.expand_dim_mappings.push_back(
          {.dim = i, .scale = target_dim});
      continue;
    }

    // if `source_dim` and `target_dim` are not equal and `source_dim` is not
    // equal to 1 then we cannot broadcast `source_shape` into the
    // `target_shape`.
    if (source_dim != target_dim) {
      return std::nullopt;
    }
  }

  std::ranges::reverse(broadcast_info.unsqueezed_shape);

  return broadcast_info;
}

static auto performBroadcasting(TraceSession& session, InstId source_id,
                                llvm::ArrayRef<i64> source_shape,
                                llvm::ArrayRef<i64> target_shape) -> InstId {
  auto broadcast_info = tryGetBroadcastInfo(source_shape, target_shape);
  if (!broadcast_info) {
    throw std::runtime_error(std::format("Failed to broadcast {} into {}.",
                                         source_shape, target_shape));
  }

  // Add unit dimensions since they are not equal
  if (target_shape.size() != source_shape.size()) {
    source_id = session.createInst<insts::Reshape>(
        source_id, broadcast_info->unsqueezed_shape);
  }

  return session.createInst<insts::ExpandDims>(
      source_id, broadcast_info->expand_dim_mappings);
}

static auto ensureHasSameDataType(Tensor& lhs, Tensor& rhs) -> void {
  if (lhs.data_type() != rhs.data_type()) {
    throw nb::value_error(
        "Cannot operate on two tensors with different data types");
  }
}

static auto ensureHasSameShape(Tensor& lhs, Tensor& rhs) -> void {
  if (lhs.shape() != rhs.shape()) {
    throw nb::value_error("Expected the two tensors to have the same shape.");
  }
}

static auto getTraceSession(std::shared_ptr<Tensor>& lhs,
                            std::shared_ptr<Tensor>& rhs)
    -> std::shared_ptr<TraceSession> {
  if (lhs->isRoot() && rhs->isRoot()) {
    auto session = std::make_shared<TraceSession>();
    session->declareParam(lhs);
    session->declareParam(rhs);
    return session;
  }

  if (lhs->isRoot() && !rhs->isRoot()) {
    auto session = rhs->session();
    session->ensureReadyForNewTrace();
    session->declareParam(lhs);
    return session;
  }

  if (!lhs->isRoot() && rhs->isRoot()) {
    auto session = lhs->session();
    session->ensureReadyForNewTrace();
    session->declareParam(rhs);
    return session;
  }

  if (!lhs->isRoot() && !rhs->isRoot()) {
    if (lhs->session() != rhs->session()) {
      auto session = lhs->session();
      // merge with the trace session from the rhs;
      session->merge(*rhs->session());
      // set the session for rhs
      rhs->session() = session;
      return session;
    }

    return lhs->session();
  }
  AXON_UNREACHABLE("This should be an unreachable point");
}

static auto getTraceSession(std::shared_ptr<Tensor>& input)
    -> std::shared_ptr<TraceSession> {
  if (input->isRoot()) {
    auto session = std::make_shared<TraceSession>();
    session->declareParam(input);
    return session;
  }

  return input->session();
}

// Compute the broadcasted target shape following NumPy/PyTorch rules.
// Align from the right; each axis must be equal or 1; result dim is max.
static auto computeBroadcastedShape(llvm::ArrayRef<i64> a,
                                    llvm::ArrayRef<i64> b)
    -> llvm::SmallVector<i64> {
  i64 ar = static_cast<i64>(a.size());
  i64 br = static_cast<i64>(b.size());
  i64 r = std::max(ar, br);
  llvm::SmallVector<i64> out;
  out.resize(r);

  for (i64 i = 0; i < r; ++i) {
    i64 ai = (i < r - ar) ? 1 : a[i - (r - ar)];
    i64 bi = (i < r - br) ? 1 : b[i - (r - br)];

    if (ai != bi && ai != 1 && bi != 1) {
      throw std::runtime_error(
          std::format("Incompatible shapes for broadcasting {} and {}", a, b));
    }
    out[i] = ai == 1 ? bi : ai;
  }
  return out;
}

export template <typename ElementWiseInst>
auto performBinaryElementWiseOperation(std::shared_ptr<Tensor> lhs,
                                       std::shared_ptr<Tensor> rhs)
    -> std::shared_ptr<Tensor> {
  ensureHasSameDataType(*lhs, *rhs);

  auto session = getTraceSession(lhs, rhs);

  auto lhs_id = session->getInstId(lhs.get());
  auto rhs_id = session->getInstId(rhs.get());

  llvm::SmallVector<i64> lhs_shape(lhs->shape());
  llvm::SmallVector<i64> rhs_shape(rhs->shape());

  if (lhs_shape == rhs_shape) {
    return session->createTensor<ElementWiseInst>(lhs_id, rhs_id);
  }

  auto target = computeBroadcastedShape(lhs_shape, rhs_shape);
  if (lhs_shape != target) {
    lhs_id = performBroadcasting(*session, lhs_id, lhs_shape, target);
  }
  if (rhs_shape != target) {
    rhs_id = performBroadcasting(*session, rhs_id, rhs_shape, target);
  }

  return session->createTensor<ElementWiseInst>(lhs_id, rhs_id);
}

export auto performMatMul(std::shared_ptr<Tensor> lhs,
                          std::shared_ptr<Tensor> rhs)
    -> std::shared_ptr<Tensor> {
  // Support vectors/2D/3D with at most one batch dim. Broadcast batch dims
  // using computeBroadcastedShape, then validate/handle the last two dims.
  if (lhs->rank() == 0 || rhs->rank() == 0) {
    throw std::runtime_error("MatMul does not support scalar operands.");
  }

  if (lhs->rank() > 3 || rhs->rank() > 3) {
    throw std::runtime_error(
        "Attempted to multiply tensors with more than rank of 3.");
  }

  ensureHasSameDataType(*lhs, *rhs);
  auto session = getTraceSession(lhs, rhs);

  llvm::SmallVector<i64> lhs_shape(session->getShape(lhs.get()));
  llvm::SmallVector<i64> rhs_shape(session->getShape(rhs.get()));

  bool lhs_is_vec = lhs_shape.size() == 1;
  bool rhs_is_vec = rhs_shape.size() == 1;

  static constexpr auto getMK = [](llvm::ArrayRef<i64> shape,
                                   bool is_vec) -> std::pair<i64, i64> {
    if (is_vec) {
      return {1, shape[0]};
    }
    AXON_DCHECK(shape.size() >= 2);
    return {shape[shape.size() - 2], shape[shape.size() - 1]};
  };

  static constexpr auto getKN = [](llvm::ArrayRef<i64> shape,
                                   bool is_vec) -> std::pair<i64, i64> {
    if (is_vec) {
      return {shape[0], 1};
    }
    AXON_DCHECK(shape.size() >= 2);
    return {shape[shape.size() - 2], shape[shape.size() - 1]};
  };

  auto [M, K_lhs] = getMK(lhs_shape, lhs_is_vec);
  auto [K_rhs, N] = getKN(rhs_shape, rhs_is_vec);
  if (K_lhs != K_rhs) {
    throw std::runtime_error(std::format(
        "Incompatible shapes for matmul {} and {} (contracted dims)", lhs_shape,
        rhs_shape));
  }

  llvm::SmallVector<i64> lhs_batch;
  llvm::SmallVector<i64> rhs_batch;
  if (!lhs_is_vec && lhs_shape.size() == 3) {
    lhs_batch.push_back(lhs_shape[0]);
  }
  if (!rhs_is_vec && rhs_shape.size() == 3) {
    rhs_batch.push_back(rhs_shape[0]);
  }

  llvm::SmallVector<i64> batch_target =
      computeBroadcastedShape(lhs_batch, rhs_batch);
  if (batch_target.size() > 1) {
    throw std::runtime_error(std::format(
        "Only up to rank-3 tensors are supported for matmul; got batch dims {}",
        batch_target));
  }

  // Build targets and broadcast both sides to targets if needed.
  llvm::SmallVector<i64> lhs_target(batch_target.begin(), batch_target.end());
  lhs_target.push_back(M);
  lhs_target.push_back(K_lhs);

  llvm::SmallVector<i64> rhs_target(batch_target.begin(), batch_target.end());
  rhs_target.push_back(K_rhs);
  rhs_target.push_back(N);

  auto lhs_id = session->getInstId(lhs.get());
  auto rhs_id = session->getInstId(rhs.get());

  if (lhs_shape != lhs_target) {
    lhs_id = performBroadcasting(*session, lhs_id, lhs_shape, lhs_target);
    lhs_shape = lhs_target;
  }
  if (rhs_shape != rhs_target) {
    rhs_id = performBroadcasting(*session, rhs_id, rhs_shape, rhs_target);
    rhs_shape = rhs_target;
  }

  // Create matmul
  InstId result_id = session->createInst<insts::MatMul>(lhs_id, rhs_id);

  // Squeeze vector results if necessary
  i64 current_rank = static_cast<i64>(batch_target.size()) + 2;
  if (lhs_is_vec && current_rank >= 2) {
    result_id = session->createInst<insts::Squeeze>(
        result_id, static_cast<i32>(current_rank - 2));
    current_rank -= 1;
  }
  if (rhs_is_vec && current_rank >= 1) {
    result_id = session->createInst<insts::Squeeze>(
        result_id, static_cast<i32>(current_rank - 1));
    current_rank -= 1;
  }

  return std::make_shared<Tensor>(session, result_id);
}

export template <Numeric T>
auto performScalarMul(std::shared_ptr<Tensor> input, T scalar_value)
    -> std::shared_ptr<Tensor> {
  auto session = getTraceSession(input);
  auto data_type = DataType::fromType<T>();

  if (data_type != input->data_type()) {
    // TODO: Improve this error message by adding a format specifier for
    // DataType and scalar.
    throw nb::value_error("Tried to multiply a tensor with different dtype");
  }

  Scalar scalar{scalar_value};
  return session->createTensor<insts::ScalarMul>(input, scalar);
}

export template <Numeric T>
auto performPow(std::shared_ptr<Tensor> input, T exponent_value)
    -> std::shared_ptr<Tensor> {
  auto session = getTraceSession(input);
  auto data_type = DataType::fromType<T>();

  if (data_type != input->data_type()) {
    throw nb::value_error("Tried to pow a tensor with different dtype");
  }

  Scalar exponent{exponent_value};
  return session->createTensor<insts::Pow>(input, exponent);
}

export template <typename InstType>
auto performReduceInst(std::shared_ptr<Tensor> input,
                       std::optional<i32> optional_axis, bool keep_dims)
    -> std::shared_ptr<Tensor> {
  auto session = getTraceSession(input);

  if (auto axis = optional_axis) {
    return session->createTensor<InstType>(input, *axis, keep_dims);
  }

  InstId reduce_id = session->getInstId(input.get());
  for (auto dim : input->shape()) {
    reduce_id =
        session->createInst<InstType>(reduce_id, 0, /*keep_dims=*/false);
  }
  return std::make_shared<Tensor>(session, reduce_id);
}

export auto performSoftmax(std::shared_ptr<Tensor> input, i32 axis)
    -> std::shared_ptr<Tensor> {
  if (axis >= input->rank()) {
    throw nb::value_error("Passed `dim` exceeded the rank of the tensor.");
  }
  auto session = getTraceSession(input);
  return session->createTensor<insts::Softmax>(input, axis);
}

export template <insts::Compare::Predicate predicate>
auto performComparison(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs)
    -> std::shared_ptr<Tensor> {
  ensureHasSameDataType(*lhs, *rhs);
  ensureHasSameShape(*lhs, *rhs);

  auto session = getTraceSession(lhs, rhs);
  return session->createTensor<insts::Compare>(lhs, rhs, predicate);
}

export auto performRelu(std::shared_ptr<Tensor> input)
    -> std::shared_ptr<Tensor> {
  auto session = getTraceSession(input);
  return session->createTensor<insts::Relu>(input);
}

export auto performAccumulate(std::shared_ptr<Tensor> sink,
                              std::shared_ptr<Tensor> source) -> void {
  auto should_emit_grad = Runtime::get().shouldEmitGrad();
  if (should_emit_grad) {
    throw nb::value_error(
        "tensor.accumulate should be performed on a no gradient context.");
  }

  auto session = getTraceSession(sink, source);
  session->createInst<insts::AccumulateData>(sink, source);
  session->evaluate();
}

}  // namespace axon
