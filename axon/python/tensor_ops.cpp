module;

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "nanobind/nanobind.h"

export module axon.python:tensor_ops;

import axon.base;
import axon.core;

import :tensor;
import :jit;

namespace nb = nanobind;

namespace axon {

auto Tensor::evaluate() -> void {
  AXON_DCHECK(context_ != nullptr);
  AXON_DCHECK(context_->insts().contains(this),
              "Context must know this tensor.");

  context_->setTensorToEvaluate(this);

  GlobalContext::get().execute(*context_);

  AXON_DCHECK(isEvaluated(), "Tensor must be evaluated at this point.");

  context_->setTensorToEvaluate(nullptr);
}

auto Tensor::backward(Tensor& grad) -> void {}

template <typename Arg>
auto getCorrespondingInstId(GraphContext& context, Arg&& arg) -> auto {
  if constexpr (std::is_same_v<Arg, std::shared_ptr<Tensor>>) {
    auto ptr = arg.get();
    return context.insts()[ptr];
  } else {
    return std::forward<Arg>(arg);
  }
}

template <typename InstType, typename... Args>
auto createTensorFromInst(std::shared_ptr<GraphContext> context, Args&&... args)
    -> std::shared_ptr<Tensor> {
  auto inst = InstType(getCorrespondingInstId(*context, args)...);
  auto inst_id = context->graph().createOp(std::move(inst), /*emit_grad=*/true);
  return std::make_shared<Tensor>(context, inst_id);
}

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

auto getOrCombineGraphContext(Tensor& lhs, Tensor& rhs)
    -> std::shared_ptr<GraphContext> {
  if (!lhs.context() && !rhs.context()) {
    auto context = std::make_shared<GraphContext>();
    lhs.declareAsParam(context);
    rhs.declareAsParam(context);

    return context;
  }

  if (!lhs.context() && rhs.context()) {
    auto context = rhs.context();
    lhs.declareAsParam(context);
    return context;
  }

  if (lhs.context() && !rhs.context()) {
    auto context = lhs.context();
    rhs.declareAsParam(context);
    return context;
  }

  if (lhs.context() && rhs.context()) {
    if (lhs.context() != rhs.context()) {
      auto context = lhs.context();

      // absorb the context from rhs
      context->absorb(*rhs.context());

      // set the context for rhs
      rhs.context() = context;
      return context;
    }

    return lhs.context();
  }

  AXON_UNREACHABLE("This should be an unreachable point");
}

export template <typename ElementWiseInst>
auto performBinaryElementWiseOperation(std::shared_ptr<Tensor> lhs,
                                       std::shared_ptr<Tensor> rhs)
    -> std::shared_ptr<Tensor> {
  auto context = getOrCombineGraphContext(*lhs, *rhs);

  auto lhs_id = context->insts()[lhs.get()];
  auto rhs_id = context->insts()[rhs.get()];

  auto lhs_shape = lhs->shape();
  auto rhs_shape = rhs->shape();

  if (lhs_shape.equals(rhs_shape)) {
    auto inst_id = context->graph().createOp(ElementWiseInst(lhs_id, rhs_id));
    return std::make_shared<Tensor>(context, inst_id);
  }

  auto lhs_elems = computeNumElems(lhs_shape);
  auto rhs_elems = computeNumElems(rhs_shape);

  if (lhs_elems < rhs_elems) {
    lhs_id =
        performBroadcasting(context->graph(), lhs_id, lhs_shape, rhs_shape);
  } else if (lhs_elems > rhs_elems) {
    rhs_id =
        performBroadcasting(context->graph(), rhs_id, rhs_shape, lhs_shape);
  }

  auto inst_id = context->graph().createOp(ElementWiseInst(lhs_id, rhs_id));
  return std::make_shared<Tensor>(context, inst_id);
}
//
// export auto performMatMul(Graph& graph, const Tensor& lhs, const Tensor& rhs)
//     -> Tensor {
//   static auto is_valid_matmul = [](llvm::ArrayRef<i64> lhs_shape,
//                                    llvm::ArrayRef<i64> rhs_shape) {
//     AXON_DCHECK(lhs_shape.size() == rhs_shape.size(),
//                 "At this point lhs and rhs must have the same rank");
//     if (lhs_shape.size() == 3) {
//       return lhs_shape[2] == rhs_shape[1];
//     }
//     if (lhs_shape.size() == 2) {
//       return lhs_shape[1] == rhs_shape[0];
//     }
//     return false;
//   };
//
//   llvm::ArrayRef<i64> lhs_shape = graph.getShape(lhs.inst_id());
//   llvm::ArrayRef<i64> rhs_shape = graph.getShape(rhs.inst_id());
//
//   auto lhs_id = lhs.inst_id();
//   auto rhs_id = rhs.inst_id();
//
//   if (lhs_shape.size() > 3 || rhs_shape.size() > 3 || lhs_shape.size() < 1 ||
//       rhs_shape.size() < 1) {
//     throw std::runtime_error(
//         "Attempted to multiply tensors with more than rank of 3.");
//   }
//
//   auto lhs_elems = computeNumElems(lhs_shape);
//   auto rhs_elems = computeNumElems(rhs_shape);
//
//   if (lhs_elems < rhs_elems) {
//     llvm::SmallVector<i64> target_shape(lhs_shape);
//
//     if (lhs_shape.size() == 2 && rhs_shape.size() == 3) {
//       target_shape.insert(target_shape.begin(), rhs_shape[0]);
//     } else if (lhs_shape.size() == 3 && rhs_shape.size() == 3) {
//       target_shape[0] = rhs_shape[0];
//     } else {
//       AXON_UNREACHABLE("TODO");
//     }
//
//     if (not is_valid_matmul(target_shape, rhs_shape)) {
//       throw std::runtime_error(
//           std::format("Cannot perform matrix multiplication on tensors with "
//                       "shape {} and {}",
//                       lhs_shape, rhs_shape));
//     }
//
//     lhs_id = performBroadcasting(graph, lhs_id, lhs_shape, target_shape);
//   }
//
//   if (lhs_elems > rhs_elems) {
//     llvm::SmallVector<i64> target_shape(rhs_shape);
//     if (lhs_shape.size() == 3 && rhs_shape.size() == 2) {
//       target_shape.insert(target_shape.begin(), lhs_shape[0]);
//     } else if (lhs_shape.size() == 3 && rhs_shape.size() == 3) {
//       target_shape[0] = rhs_shape[0];
//     } else {
//       AXON_UNREACHABLE("TODO");
//     }
//
//     if (not is_valid_matmul(lhs_shape, target_shape)) {
//       throw std::runtime_error(
//           std::format("Cannot perform matrix multiplication on tensors with "
//                       "shape {} and {}",
//                       lhs_shape, rhs_shape));
//     }
//
//     rhs_id = performBroadcasting(graph, rhs_id, rhs_shape, target_shape);
//   }
//
//   return Tensor(graph.createOp(insts::MatMul(lhs_id, rhs_id)));
// }

}  // namespace axon
