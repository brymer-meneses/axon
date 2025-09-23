module;

#include <algorithm>
#include <format>
#include <memory>
#include <optional>
#include <utility>

#include "axon/base/macros.h"
#include "llvm/ADT/SmallVector.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"

export module axon.python:tensor;

import axon.base;
import axon.core;
import axon.mlir;

namespace nb = nanobind;

namespace axon {

export class TraceSession;

export class Tensor {
 public:
  Tensor(Storage&& storage, bool requires_grad)
      : storage_(std::make_unique<Storage>(std::move(storage))),
        requires_grad_(requires_grad) {}

  Tensor(std::shared_ptr<TraceSession> session, InstId inst_id);

  ~Tensor();

  auto rank() const -> u64 { return shape().size(); }

  auto zeroGrad() -> void;

  auto shape() const -> llvm::ArrayRef<i64>;

  auto asString() -> std::string;

  auto isEvaluated() const -> bool { return storage_ != nullptr; }

  auto isRoot() const -> bool;

  auto requiresGrad() const -> bool { return requires_grad_; }

  auto evaluate() -> void;
  auto backward(std::shared_ptr<Tensor> grad) -> void;

  auto grad() const -> std::shared_ptr<Tensor> { return grad_; }

  auto data_type() const -> DataType;

  auto storage() -> Storage* { return storage_.get(); }
  auto storage() const -> const Storage* { return storage_.get(); }

  auto session() -> std::shared_ptr<TraceSession>& { return session_; }
  auto session() const -> const std::shared_ptr<TraceSession>& {
    return session_;
  }

  auto setStorage(std::unique_ptr<Storage> storage) -> void {
    storage_ = std::move(storage);
  }

 private:
  std::unique_ptr<Storage> storage_;
  std::shared_ptr<TraceSession> session_;
  std::shared_ptr<Tensor> grad_;

  bool requires_grad_;
};

}  // namespace axon
