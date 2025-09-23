module;

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

static auto performBroadcasting(Graph& graph, InstId source_id,
                                llvm::ArrayRef<i64> source_shape,
                                llvm::ArrayRef<i64> target_shape) -> InstId {
  auto broadcast_info = tryGetBroadcastInfo(source_shape, target_shape);
  if (!broadcast_info) {
    throw std::runtime_error(std::format("Failed to broadcast {} into {}.",
                                         source_shape, target_shape));
  }

  // Add unit dimensions since they are not equal
  if (target_shape.size() != source_shape.size()) {
    source_id = graph.createOp(
        insts::Reshape(source_id, broadcast_info->unsqueezed_shape));
  }

  return graph.createOp(
      insts::ExpandDims(source_id, broadcast_info->expand_dim_mappings));
}

static auto validateDataType(Tensor& lhs, Tensor& rhs) -> void {
  if (lhs.data_type() != rhs.data_type()) {
    throw nb::value_error(
        "Cannot operate on two tensors with different data types");
  }
}

static auto getTraceSession(Tensor& lhs, Tensor& rhs)
    -> std::shared_ptr<TraceSession> {
  if (lhs.isRoot() && rhs.isRoot()) {
    auto session = std::make_shared<TraceSession>();
    session->declareParam(&lhs);
    session->declareParam(&rhs);
    return session;
  }

  if (lhs.isRoot() && !rhs.isRoot()) {
    auto session = rhs.session();
    session->declareParam(&lhs);
    return session;
  }

  if (!lhs.isRoot() && rhs.isRoot()) {
    auto session = lhs.session();
    session->declareParam(&rhs);
    return session;
  }

  if (!lhs.isRoot() && !rhs.isRoot()) {
    if (lhs.session() != rhs.session()) {
      auto session = lhs.session();
      // merge with the trace session from the rhs;
      session->merge(*rhs.session());
      // set the session for rhs
      rhs.session() = session;
      return session;
    }

    return lhs.session();
  }
  AXON_UNREACHABLE("This should be an unreachable point");
}

static auto getTraceSession(Tensor& input) -> std::shared_ptr<TraceSession> {
  if (input.isRoot()) {
    auto session = std::make_shared<TraceSession>();
    session->declareParam(&input);
    return session;
  }

  return input.session();
}

export template <typename ElementWiseInst>
auto performBinaryElementWiseOperation(std::shared_ptr<Tensor> lhs,
                                       std::shared_ptr<Tensor> rhs)
    -> std::shared_ptr<Tensor> {
  validateDataType(*lhs, *rhs);

  auto session = getTraceSession(*lhs, *rhs);

  auto lhs_id = session->insts()[lhs.get()];
  auto rhs_id = session->insts()[rhs.get()];

  auto lhs_shape = lhs->shape();
  auto rhs_shape = rhs->shape();

  if (lhs_shape.equals(rhs_shape)) {
    auto inst_id = session->graph().createOp(ElementWiseInst(lhs_id, rhs_id));
    return std::make_shared<Tensor>(session, inst_id);
  }

  auto lhs_elems = computeNumElems(lhs_shape);
  auto rhs_elems = computeNumElems(rhs_shape);

  if (lhs_elems < rhs_elems) {
    lhs_id =
        performBroadcasting(session->graph(), lhs_id, lhs_shape, rhs_shape);
  } else if (lhs_elems > rhs_elems) {
    rhs_id =
        performBroadcasting(session->graph(), rhs_id, rhs_shape, lhs_shape);
  }

  auto inst_id = session->graph().createOp(ElementWiseInst(lhs_id, rhs_id));
  return std::make_shared<Tensor>(session, inst_id);
}

export auto performMatMul(std::shared_ptr<Tensor> lhs,
                          std::shared_ptr<Tensor> rhs)
    -> std::shared_ptr<Tensor> {
  static constexpr auto is_valid_matmul = [](llvm::ArrayRef<i64> lhs_shape,
                                             llvm::ArrayRef<i64> rhs_shape) {
    AXON_DCHECK(lhs_shape.size() == rhs_shape.size(),
                "At this point lhs and rhs must have the same rank");
    if (lhs_shape.size() == 3) {
      return lhs_shape[2] == rhs_shape[1];
    }
    if (lhs_shape.size() == 2) {
      return lhs_shape[1] == rhs_shape[0];
    }
    return false;
  };

  validateDataType(*lhs, *rhs);
  auto session = getTraceSession(*lhs, *rhs);
  auto& graph = session->graph();

  llvm::ArrayRef<i64> lhs_shape = session->getShape(lhs.get());
  llvm::ArrayRef<i64> rhs_shape = session->getShape(rhs.get());

  auto lhs_id = session->insts()[lhs.get()];
  auto rhs_id = session->insts()[rhs.get()];

  if (lhs_shape.size() > 3 || rhs_shape.size() > 3 || lhs_shape.size() < 1 ||
      rhs_shape.size() < 1) {
    throw std::runtime_error(
        "Attempted to multiply tensors with more than rank of 3.");
  }

  auto lhs_elems = computeNumElems(lhs_shape);
  auto rhs_elems = computeNumElems(rhs_shape);

  if (lhs_elems < rhs_elems) {
    llvm::SmallVector<i64> target_shape(lhs_shape);

    if (lhs_shape.size() == 2 && rhs_shape.size() == 3) {
      target_shape.insert(target_shape.begin(), rhs_shape[0]);
    } else if (lhs_shape.size() == 3 && rhs_shape.size() == 3) {
      target_shape[0] = rhs_shape[0];
    }

    if (not is_valid_matmul(target_shape, rhs_shape)) {
      throw std::runtime_error(
          std::format("Cannot perform matrix multiplication on tensors with "
                      "shape {} and {}",
                      lhs_shape, rhs_shape));
    }

    lhs_id = performBroadcasting(graph, lhs_id, lhs_shape, target_shape);
  }

  if (lhs_elems > rhs_elems) {
    llvm::SmallVector<i64> target_shape(rhs_shape);
    if (lhs_shape.size() == 3 && rhs_shape.size() == 2) {
      target_shape.insert(target_shape.begin(), lhs_shape[0]);
    } else if (lhs_shape.size() == 3 && rhs_shape.size() == 3) {
      target_shape[0] = rhs_shape[0];
    }

    if (not is_valid_matmul(lhs_shape, target_shape)) {
      throw std::runtime_error(
          std::format("Cannot perform matrix multiplication on tensors with "
                      "shape {} and {}",
                      lhs_shape, rhs_shape));
    }

    rhs_id = performBroadcasting(graph, rhs_id, rhs_shape, target_shape);
  }

  auto inst_id = graph.createOp(insts::MatMul(lhs_id, rhs_id));
  return std::make_shared<Tensor>(session, inst_id);
}

export template <Numeric T>
auto performScalarMul(std::shared_ptr<Tensor> input, T scalar_value)
    -> std::shared_ptr<Tensor> {
  auto session = getTraceSession(*input);
  auto data_type = DataType::fromType<T>();

  if (data_type != input->data_type()) {
    // TODO: Improve this error message by adding a format specifier for
    // DataType and scalar.
    throw nb::value_error("Tried to multiply a tensor with different dtype");
  }

  Scalar scalar{scalar_value};
  auto input_id = session->insts().at(input.get());
  auto product_inst_id =
      session->graph().createOp(insts::ScalarMul(input_id, scalar));
  return std::make_shared<Tensor>(session, product_inst_id);
}

export template <typename InstType>
auto performReduceInst(std::shared_ptr<Tensor> input, i32 axis, bool keep_dims)
    -> std::shared_ptr<Tensor> {
  auto session = getTraceSession(*input);

  auto& graph = session->graph();
  auto input_inst_id = session->insts()[input.get()];
  auto softmax_inst_id =
      graph.createOp(InstType(input_inst_id, axis, keep_dims));

  return std::make_shared<Tensor>(session, softmax_inst_id);
}

export auto performSoftmax(std::shared_ptr<Tensor> input, i32 axis)
    -> std::shared_ptr<Tensor> {
  if (axis >= input->rank()) {
    throw nb::value_error("Passed `dim` exceeded the rank of the tensor.");
  }
  auto session = getTraceSession(*input);

  auto& graph = session->graph();
  auto input_inst_id = session->insts()[input.get()];
  auto softmax_inst_id = graph.createOp(insts::Softmax(input_inst_id, axis));

  return std::make_shared<Tensor>(session, softmax_inst_id);
}

}  // namespace axon
