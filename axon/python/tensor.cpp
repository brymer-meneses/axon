module;

#include <iomanip>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <utility>

#include "axon/base/macros.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_os_ostream.h"
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

export class Tensor : nb::intrusive_base {
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

  auto shape() const -> llvm::ArrayRef<int64_t> {
    if (hasData()) {
      return data_->shape();
    }

    return {};
  }

  auto rank() const -> int64_t {
    if (hasData()) {
      return static_cast<int64_t>(data_->shape().size());
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
                          size_t dim, llvm::ArrayRef<int64_t> shape,
                          llvm::ArrayRef<int64_t> strides, int indent_width,
                          int depth) -> void {
  if (dim == shape.size() - 1) {
    // Base case: 1-D row
    stream << "[";
    for (int64_t i = 0; i < shape[dim]; ++i) {
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
    for (int64_t i = 0; i < shape[dim]; ++i) {
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

}  // namespace axon
