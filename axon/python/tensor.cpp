module;

#include <format>
#include <memory>
#include <optional>
#include <utility>

#include "axon/base/macros.h"
#include "llvm/ADT/DenseMap.h"
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

template <typename T>
concept NumericType = std::integral<T> || std::floating_point<T>;

template <NumericType T>
static auto dumpRecursive(llvm::raw_string_ostream& stream, const T* ptr,
                          size_t dim, llvm::ArrayRef<i64> shape,
                          llvm::ArrayRef<i64> strides, int indent_width,
                          int depth = 0) -> void {
  static constexpr auto dump_formatted = [](llvm::raw_string_ostream& stream,
                                            T elem) {
    if constexpr (std::is_floating_point_v<T>) {
      stream << std::format("{:.4f}", elem);
    } else if constexpr (std::is_integral_v<T>) {
      stream << elem;
    }
  };

  if (dim == shape.size() - 1) {
    // Base case: 1-D row
    stream << "[";
    for (i64 i = 0; i < shape[dim]; ++i) {
      dump_formatted(stream, ptr[i * strides[dim]]);

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
        for (i64 j = 0; j < (depth + 1) * indent_width; ++j) {
          stream << ' ';
        }
      }
      dumpRecursive(stream, ptr + i * strides[dim], dim + 1, shape, strides,
                    indent_width, depth + 1);
    }
    stream << "]";
  }
}

export class Tensor;

export class GraphContext {
 public:
  auto getShape(const Tensor* tensor) const -> llvm::ArrayRef<i64> {
    AXON_DCHECK(insts_.contains(tensor));

    auto inst_id = insts_.at(tensor);
    return graph_.getShape(inst_id);
  }

  auto absorb(GraphContext& other) -> void {
    auto offset = graph_.insts().size();

    for (auto [tensor, inst_id] : other.insts_) {
      insts_[tensor] = InstId(static_cast<i32>(offset) + inst_id.value());
    }

    for (auto tensor : other.parameters_) {
      parameters_.push_back(tensor);
    }

    graph_.absorb(other.graph_);
  }

  auto graph() -> Graph& { return graph_; }
  auto graph() const -> const Graph& { return graph_; }

  auto insts() -> llvm::DenseMap<Tensor*, InstId>& { return insts_; }
  auto insts() const -> const llvm::DenseMap<Tensor*, InstId>& {
    return insts_;
  }

  auto parameters() -> llvm::SmallVector<Tensor*> { return parameters_; }
  auto parameters() const -> const llvm::SmallVector<Tensor*>& {
    return parameters_;
  }

  auto getTensorToEvaluate() const -> Tensor* { return to_evaluate_; }

  auto setTensorToEvaluate(Tensor* tensor) -> void {
    if (tensor == nullptr) {
      graph_.setReturned(InstId::None);
      return;
    }

    AXON_DCHECK(insts_.contains(tensor));

    graph_.setReturned(insts_[tensor]);
    to_evaluate_ = tensor;
  }

 private:
  Graph graph_;
  Tensor* to_evaluate_;
  llvm::DenseMap<Tensor*, InstId> insts_;
  llvm::SmallVector<Tensor*> parameters_;
};

export class Tensor {
 public:
  Tensor(Storage&& storage, bool requires_grad)
      : storage_(std::make_unique<Storage>(std::move(storage))),
        requires_grad_(requires_grad) {}

  Tensor(std::shared_ptr<GraphContext> context, InstId inst_id)
      : context_(std::move(context)) {
    context_->insts()[this] = inst_id;
  }

  auto zeroGrad() -> void {
    if (not requires_grad_) {
      throw std::runtime_error(
          "Cannot zero a tensor that does not require gradients.");
    }

    if (grad_ != nullptr) {
      grad_->storage().fillWithZeros();
    } else {
      grad_ = std::make_shared<Tensor>(Storage::createZerosLike(*storage_),
                                       /*requires_grad=*/false);
    }
  }

  auto rank() const -> u64 { return shape().size(); }

  auto shape() const -> llvm::ArrayRef<i64> {
    if (storage_) {
      return storage_->shape();
    }

    AXON_DCHECK(context_ != nullptr);
    return context_->getShape(this);
  }

  auto asString() -> std::string {
    if (!isEvaluated()) {
      evaluate();
    }

    std::string repr;
    llvm::raw_string_ostream stream{repr};

    auto shape = storage_->shape();
    auto strides = storage_->strides();

    stream << "tensor(";
    switch (storage_->data_type().kind()) {
      case DataType::Float64: {
        auto base = reinterpret_cast<const f64*>(storage_->data_ptr());
        dumpRecursive<f64>(stream, base, 0, shape, strides, /*indent_width=*/8);
        break;
      }
      case DataType::Float32: {
        auto base = reinterpret_cast<const f32*>(storage_->data_ptr());
        dumpRecursive<f32>(stream, base, 0, shape, strides, /*indent_width=*/8);
        break;
      }
    }

    if (requires_grad_) {
      stream << ", requires_grad=True";
    }
    stream << ")";

    stream.flush();
    return repr;
  }

  auto isEvaluated() const -> bool { return storage_ != nullptr; }

  auto requiresGrad() const -> bool { return requires_grad_; }

  auto grad() const -> std::shared_ptr<Tensor> { return grad_; }

  auto storage() -> Storage& { return *storage_; }
  auto storage() const -> const Storage& { return *storage_; }

  auto context() -> std::shared_ptr<GraphContext> { return context_; }
  auto context() const -> const std::shared_ptr<GraphContext> {
    return context_;
  }

  auto declareAsParam(std::shared_ptr<GraphContext> context) -> void {
    context_ = std::move(context);

    auto inst_id =
        context_->graph().declareParam(storage_->shape(), requires_grad_);

    context_->insts()[this] = inst_id;
    context_->parameters().push_back(this);
  }

  // defined in tensor_ops.cpp
  auto evaluate() -> void;
  auto backward(Tensor& grad) -> void;

 private:
  std::unique_ptr<Storage> storage_;
  std::shared_ptr<GraphContext> context_;
  std::shared_ptr<Tensor> grad_;

  bool requires_grad_;
};

}  // namespace axon
