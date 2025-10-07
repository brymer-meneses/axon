module;

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>

#include "axon/base/macros.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "nanobind/nanobind.h"

export module axon.python:tensor_impl;

import axon.base;
import axon.core;

import :tensor;
import :runtime;
import :jit;
import :storage;
import :trace_session;

namespace axon {

Tensor::~Tensor() {
  if (session_) {
    session_->insts().erase(this);
    auto& params = session_->parameters();
    params.erase(std::remove_if(params.begin(), params.end(),
                                [&](auto& p) { return p.get() == this; }),
                 params.end());
  }
}

auto Tensor::shape() const -> llvm::ArrayRef<i64> {
  if (storage_) {
    return storage_->shape();
  }

  // This is a lazy tensor, so get the shape from the trace session.
  AXON_ASSERT(session_ != nullptr);
  return session_->getShape(this);
}

Tensor::Tensor(std::shared_ptr<TraceSession> session, InstId inst_id)
    : session_(std::move(session)) {
  session_->insts()[this] = inst_id;
  requires_grad_ = session_->checkRequiresGrad(inst_id);
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
    auto cpu = CpuStorage::createZerosLike(*storage_);
    grad_ = std::make_shared<Tensor>(Storage(std::move(cpu)),
                                     /*requires_grad=*/false);
  }
}

auto Tensor::save() -> void {
  if (!session_) {
    throw nb::value_error("Tried to save a root tensor.");
  }

  session_->markAsSaved(this);
}

auto Tensor::evaluate() -> void {
  if (!session_) {
    throw nb::value_error("Tried to evaluate an already materialized tensor.");
  }

  session_->evaluate(this);
  session_->markForReset();
}

auto Tensor::backward(std::shared_ptr<Tensor> grad) -> void {
  AXON_DCHECK(session_ != nullptr);

  if (!grad) {
    if (rank() != 0) {
      throw nb::value_error(
          "Only scalar tensors can have their gradient inferred.");
    }
    session_->performBackward(this, grad.get());
    return;
  }

  if (grad->data_type() != data_type()) {
    throw std::runtime_error(
        std::format("Received gradient has different dtype with this tensor, "
                    "expected {} got {}",
                    data_type(), grad->data_type()));
  }

  if (!session_->insts().contains(grad.get())) {
    session_->declareParam(grad);
  }
  session_->performBackward(this, grad.get());
}

auto Tensor::getInstId() const -> InstId {
  if (session_) {
    return session_->getInstId(this);
  }

  return InstId::None;
}

auto Tensor::isRoot() const -> bool {
  if (session_ != nullptr) {
    return session_->parameters().size() == 0;
  }
  return true;
}

template <Numeric T>
static auto dumpRecursive(llvm::raw_string_ostream& stream,
                          const Storage* storage, size_t dim,
                          llvm::SmallVector<i64>& idx, int indent_width,
                          int depth = 0) -> void {
  auto shape = storage->shape();
  static constexpr auto dump_formatted = [](llvm::raw_string_ostream& stream,
                                            T elem) {
    if constexpr (std::is_same_v<T, bool>) {
      stream << (elem ? "True" : "False");
    } else if constexpr (std::is_floating_point_v<T>) {
      stream << llvm::formatv("{0:F4}", elem);
    } else if constexpr (std::is_integral_v<T>) {
      stream << elem;
    }
  };

  // Handle scalar (rank-0) tensors.
  if (shape.empty()) {
    dump_formatted(stream, storage->getElementAt<T>({}));
    return;
  }

  if (dim == shape.size() - 1) {
    // Base case: 1-D row
    stream << "[";
    for (i64 i = 0; i < shape[dim]; ++i) {
      idx[dim] = i;
      dump_formatted(stream, storage->getElementAt<T>(idx));

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
      idx[dim] = i;
      dumpRecursive<T>(stream, storage, dim + 1, idx, indent_width, depth + 1);
    }
    stream << "]";
  }
}

auto Tensor::asString() -> std::string {
  if (!isEvaluated()) {
    std::string repr;
    llvm::raw_string_ostream stream{repr};
    stream << "LazyTensor(";
    if (requires_grad_) {
      stream << "requires_grad)";
    }
    return repr;
  }

  std::string repr;
  llvm::raw_string_ostream stream{repr};

  auto shape = storage_->shape();
  llvm::SmallVector<i64> idx(shape.size(), 0);

  stream << "tensor(";
  switch (storage_->data_type().kind()) {
    case DataType::Float64: {
      dumpRecursive<f64>(stream, storage_.get(), 0, idx,
                         /*indent_width=*/8);
      break;
    }
    case DataType::Float32: {
      dumpRecursive<f32>(stream, storage_.get(), 0, idx,
                         /*indent_width=*/8);
      break;
    }
    case DataType::Int1: {
      dumpRecursive<bool>(stream, storage_.get(), 0, idx,
                          /*indent_width=*/8);
      break;
    }
    case DataType::Int32: {
      dumpRecursive<i32>(stream, storage_.get(), 0, idx,
                         /*indent_width=*/8);
      break;
    }
    case DataType::Int64: {
      dumpRecursive<i64>(stream, storage_.get(), 0, idx,
                         /*indent_width=*/8);
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
