module;

#include <algorithm>
#include <memory>

#include "axon/base/macros.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "nanobind/nanobind.h"

export module axon.python:tensor_impl;

import axon.base;
import axon.core;

import :tensor;

import :jit;

namespace axon {

export class TraceSession : public std::enable_shared_from_this<TraceSession> {
 public:
  auto getInstId(const Tensor* tensor) const -> InstId {
    AXON_DCHECK(insts_.contains(tensor));
    return insts_.at(tensor);
  }
  auto getShape(const Tensor* tensor) const -> llvm::ArrayRef<i64> {
    auto inst_id = getInstId(tensor);
    return graph_.getShape(inst_id);
  }

  auto getDataType(const Tensor* tensor) const -> DataType {
    auto inst_id = getInstId(tensor);
    return graph_.getDataType(inst_id);
  }

  auto declareParam(const Tensor* tensor) -> void {
    AXON_DCHECK(tensor->isEvaluated());

    auto storage = tensor->storage();

    auto inst_id = graph_.declareParam(storage->shape(), storage->data_type(),
                                       tensor->requiresGrad());

    insts_[const_cast<Tensor*>(tensor)] = inst_id;
    parameters_.push_back(const_cast<Tensor*>(tensor));
  }

  auto merge(TraceSession& other) -> void {
    auto offset = graph_.insts().size();

    for (auto [tensor, inst_id] : other.insts_) {
      insts_[tensor] = InstId(static_cast<i32>(offset) + inst_id.value());
    }

    for (auto tensor : other.parameters_) {
      parameters_.push_back(tensor);
    }

    graph_.merge(other.graph_);
  }

  auto evaluate(Tensor* tensor) -> void {
    AXON_DCHECK(insts_.contains(tensor));
    auto tensor_inst_id = insts_[tensor];

    graph_.setReturned(tensor_inst_id);
    auto storage = GlobalContext::get().execute(parameters_, tensor, graph_);
    graph_.setReturned(InstId::None);

    tensor->setStorage(std::move(storage));
  }

  auto reset() -> void {
    auto keep_alive = shared_from_this();

    for (auto& [tensor, _] : insts_) {
      if (tensor->session() != nullptr) {
        tensor->session() = nullptr;
      }
    }

    insts_.clear();
    parameters_.clear();
  }

  auto graph() -> Graph& { return graph_; }
  auto graph() const -> const Graph& { return graph_; }

  auto insts() -> llvm::DenseMap<Tensor*, InstId>& { return insts_; }
  auto insts() const -> const llvm::DenseMap<Tensor*, InstId>& {
    return insts_;
  }

  auto parameters() -> llvm::SmallVector<Tensor*>& { return parameters_; }
  auto parameters() const -> const llvm::SmallVector<Tensor*>& {
    return parameters_;
  }

 private:
  Graph graph_;
  llvm::DenseMap<Tensor*, InstId> insts_;
  llvm::SmallVector<Tensor*> parameters_;
};

Tensor::~Tensor() {
  if (session_) {
    session_->insts().erase(this);
    auto& params = session_->parameters();
    params.erase(std::remove(params.begin(), params.end(), this), params.end());
  }
}

auto Tensor::shape() const -> llvm::ArrayRef<i64> {
  if (storage_) {
    return storage_->shape();
  }

  // This is a lazy tensor, so get the shape from the trace session.
  AXON_DCHECK(session_ != nullptr);
  return session_->getShape(this);
}

Tensor::Tensor(std::shared_ptr<TraceSession> session, InstId inst_id)
    : session_(std::move(session)) {
  session_->insts()[this] = inst_id;
  requires_grad_ = session_->graph().checkRequiresGrad(inst_id);
}

auto Tensor::data_type() const -> DataType {
  if (storage_) {
    return storage_->data_type();
  }

  return session_->getDataType(const_cast<Tensor*>(this));
}

auto Tensor::zeroGrad() -> void {
  if (not requires_grad_) {
    throw std::runtime_error(
        "Cannot zero a tensor that does not require gradients.");
  }

  if (grad_ != nullptr) {
    grad_->storage()->fillWithZeros();
  } else {
    grad_ = std::make_shared<Tensor>(Storage::createZerosLike(*storage_),
                                     /*requires_grad=*/false);
  }
}

auto Tensor::evaluate() -> void {
  session_->evaluate(this);

  // If this tensor doesn't require gradients, then we can prune this graph.
  if (!requires_grad_) {
    session_->reset();
  }
}

auto Tensor::backward(std::shared_ptr<Tensor> grad) -> void {
  AXON_DCHECK(session_ != nullptr);

  if (!grad) {
    // we should make a scalar tensor and propagate that instead.
    AXON_UNREACHABLE("TODO");
  }

  if (grad->data_type() != data_type()) {
    throw nb::value_error(
        "Received gradient has different dtype with this tensor");
  }

  if (!grad && rank() != 0) {
    throw nb::value_error(
        "Only scalar tensors can have their gradient inferred.");
  }
  if (!session_->insts().contains(grad.get())) {
    session_->declareParam(grad.get());
  }

  auto& graph = session_->graph();
  auto grad_id = session_->getInstId(grad.get());
  auto tensor_id = session_->getInstId(this);

  axon::backward(graph, tensor_id, grad_id);

  session_->evaluate(this);
  session_->reset();
}

auto Tensor::isRoot() const -> bool {
  if (session_ != nullptr) {
    return session_->parameters().size() == 0;
  }
  return true;
}

template <Numeric T>
static auto dumpRecursive(llvm::raw_string_ostream& stream, const T* ptr,
                          size_t dim, llvm::ArrayRef<i64> shape,
                          llvm::ArrayRef<i64> strides, int indent_width,
                          int depth = 0) -> void {
  static constexpr auto dump_formatted = [](llvm::raw_string_ostream& stream,
                                            T elem) {
    if constexpr (std::is_floating_point_v<T>) {
      stream << llvm::formatv("{0:F4}", elem);
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

auto Tensor::asString() -> std::string {
  if (!isEvaluated()) {
    evaluate();
  }

  AXON_DCHECK(isEvaluated());

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

}  // namespace axon
