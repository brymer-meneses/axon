module;

#include <format>
#include <memory>
#include <optional>
#include <utility>

#include "axon/base/macros.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "nanobind/intrusive/counter.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xtensor/containers/xarray.hpp"

export module axon.python:tensor;

import axon.base;
import axon.core;
import axon.mlir;

namespace nb = nanobind;

namespace axon {

export class Tensor : public nb::intrusive_base {
 public:
  Tensor(Storage&& data, bool requires_grad)
      : data_(std::move(data)), requires_grad_(requires_grad) {}

  explicit Tensor(InstId inst_id) : inst_id_(inst_id) {}

  auto hasData() const -> bool { return data_.has_value(); }
  auto hasGrad() const -> bool { return grad_ != nullptr; }

  auto zeroGrad() -> void {
    if (not requires_grad_) {
      throw std::runtime_error(
          "Cannot zero a tensor that does not require gradients.");
    }

    if (grad_ != nullptr) {
      grad_->data()->fillWithZeros();
    } else {
      grad_ = std::make_shared<Tensor>(Storage::createZerosLike(*data_),
                                       /*requires_grad=*/false);
    }
  }

  auto trace(Graph& graph) -> void {
    AXON_DCHECK(not inst_id_.isValid(), "inst_id_ must not be valid.");
    inst_id_ = graph.declareParam(data_->shape(), requires_grad_);
    if (requires_grad_ && not hasGrad()) {
      zeroGrad();
    }
  }

  auto untrace() -> void { inst_id_ = InstId::None; }

  auto requiresGrad() const -> bool { return requires_grad_; }

  auto shape() const -> llvm::ArrayRef<i64> {
    if (hasData()) {
      return data_->shape();
    }

    return {};
  }

  auto rank() const -> i64 {
    if (hasData()) {
      return static_cast<i64>(data_->shape().size());
    }
    AXON_UNREACHABLE("TODO!");
  }

  auto inst_id() const -> InstId { return inst_id_; }

  auto data() -> std::optional<Storage>& { return data_; }
  auto data() const -> const std::optional<Storage>& { return data_; }

  auto grad() const -> std::shared_ptr<Tensor> { return grad_; }

 private:
  bool requires_grad_;

  std::optional<Storage> data_;
  std::shared_ptr<Tensor> grad_ = nullptr;

  InstId inst_id_ = InstId::None;
};

static auto dumpRecursive(llvm::raw_string_ostream& stream, const float* ptr,
                          size_t dim, llvm::ArrayRef<i64> shape,
                          llvm::ArrayRef<i64> strides, int indent_width,
                          int depth) -> void {
  if (dim == shape.size() - 1) {
    // Base case: 1-D row
    stream << "[";
    for (i64 i = 0; i < shape[dim]; ++i) {
      float elem = ptr[i * strides[dim]];
      stream << elem;
      if (i + 1 < shape[dim]) {
        stream << ", ";
      }
    }
    stream << "]";
  } else {
    // Recursive case: nested arrays
    stream << "[";
    for (i64 i = 0; i < shape[dim]; ++i) {
      if (i > 0) {
        stream << ",\n";
        for (int j = 0; j < (depth + 1) * indent_width; ++j) {
          stream << ' ';
        }
      }
      dumpRecursive(stream, ptr + i * strides[dim], dim + 1, shape, strides,
                    indent_width, depth + 1);
    }
    stream << "]";
  }
}

export auto dumpTensor(const Tensor& tensor, int indent_width = 8)
    -> std::string {
  AXON_DCHECK(tensor.hasData(), "Passed tensor must have data.");

  std::string repr;
  llvm::raw_string_ostream stream{repr};

  auto shape = tensor.shape();
  auto strides = tensor.data()->strides();
  auto base = reinterpret_cast<const float*>(tensor.data()->data_ptr());

  stream << "tensor(";
  dumpRecursive(stream, base, 0, shape, strides, indent_width, /*depth=*/0);

  if (tensor.requiresGrad()) {
    stream << ", requires_grad=True";
  }
  stream << ")";

  stream.flush();
  return repr;
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

static auto computeNumElems(llvm::ArrayRef<i64> shape) -> i64 {
  i64 num_elems = 1;
  for (auto dim : shape) {
    num_elems *= dim;
  }
  return num_elems;
}

export template <typename ElementWiseInst>
auto performElementWiseOperation(Graph& graph, const Tensor& lhs,
                                 const Tensor& rhs) -> Tensor {
  auto lhs_id = lhs.inst_id();
  auto rhs_id = rhs.inst_id();

  auto lhs_shape = graph.getShape(lhs_id);
  auto rhs_shape = graph.getShape(rhs_id);

  if (lhs_shape.equals(rhs_shape)) {
    auto inst_id = graph.createOp(ElementWiseInst(lhs_id, rhs_id));
    return Tensor(inst_id);
  }

  auto lhs_elems = computeNumElems(lhs_shape);
  auto rhs_elems = computeNumElems(rhs_shape);

  if (lhs_elems < rhs_elems) {
    lhs_id = performBroadcasting(graph, lhs_id, lhs_shape, rhs_shape);
  } else if (lhs_elems > rhs_elems) {
    rhs_id = performBroadcasting(graph, rhs_id, rhs_shape, lhs_shape);
  }

  auto inst_id = graph.createOp(ElementWiseInst(lhs_id, rhs_id));
  return Tensor(inst_id);
}

export auto performMatMul(Graph& graph, const Tensor& lhs, const Tensor& rhs)
    -> Tensor {
  static auto is_valid_matmul = [](llvm::ArrayRef<i64> lhs_shape,
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

  llvm::ArrayRef<i64> lhs_shape = graph.getShape(lhs.inst_id());
  llvm::ArrayRef<i64> rhs_shape = graph.getShape(rhs.inst_id());

  auto lhs_id = lhs.inst_id();
  auto rhs_id = rhs.inst_id();

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
    } else {
      AXON_UNREACHABLE("TODO");
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
    } else {
      AXON_UNREACHABLE("TODO");
    }

    if (not is_valid_matmul(lhs_shape, target_shape)) {
      throw std::runtime_error(
          std::format("Cannot perform matrix multiplication on tensors with "
                      "shape {} and {}",
                      lhs_shape, rhs_shape));
    }

    rhs_id = performBroadcasting(graph, rhs_id, rhs_shape, target_shape);
  }

  return Tensor(graph.createOp(insts::MatMul(lhs_id, rhs_id)));
}

}  // namespace axon
