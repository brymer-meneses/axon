module;

#include <iomanip>
#include <memory>
#include <optional>
#include <print>
#include <ranges>
#include <sstream>
#include <utility>

#include "axon/base/dcheck.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_os_ostream.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xtensor/containers/xarray.hpp"

export module axon.python:tensor;

import axon.base;
import axon.core;
import axon.mlir;

namespace nb = nanobind;

namespace axon {

export class Tensor {
 public:
  Tensor(Storage& data) : data_(std::move(data)) {}

  Tensor(Storage& data, Storage& grad)
      : data_(std::move(data)), grad_(std::make_shared<Tensor>(grad)) {}

  Tensor(InstId inst_id) : inst_id_(inst_id) {}

  auto hasData() const -> bool { return data_.has_value(); }
  auto setInstId(InstId inst_id) -> void { inst_id_ = inst_id; }

  auto inst_id() const -> InstId { return inst_id_; }

  auto requiresGrad() const -> bool { return grad_ != nullptr; }

  auto shape() const -> llvm::ArrayRef<int64_t> {
    if (hasData()) {
      return data_->shape();
    }
    std::terminate();
  }

  auto rank() const -> int64_t {
    if (hasData()) {
      return static_cast<int64_t>(data_->shape().size());
    }
    std::terminate();
  }

  auto data() -> std::optional<Storage>& { return data_; }
  auto data() const -> const std::optional<Storage>& { return data_; }

  auto grad() const -> std::shared_ptr<Tensor> { return grad_; }

 private:
  std::optional<Storage> data_;
  std::shared_ptr<Tensor> grad_;
  InstId inst_id_ = InstId::None;
};

static void dumpRecursive(llvm::raw_string_ostream& stream, const float* ptr,
                          size_t dim, llvm::ArrayRef<int64_t> shape,
                          llvm::ArrayRef<int64_t> strides, int indent_width,
                          int depth = 0) {
  if (dim == shape.size() - 1) {
    // Base case: 1-D row
    stream << "[";
    for (int64_t i = 0; i < shape[dim]; ++i) {
      stream << ptr[i * strides[dim]];
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
  dumpRecursive(stream, base, 0, shape, strides, indent_width);

  if (tensor.requiresGrad()) {
    stream << ", requires_grad=True";
  }
  stream << ")";

  stream.flush();
  return repr;
}

}  // namespace axon
