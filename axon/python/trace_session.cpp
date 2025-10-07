module;

#include <memory>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

export module axon.python:trace_session;

import :tensor;
import :runtime;

namespace axon {

export class TraceSession : public std::enable_shared_from_this<TraceSession> {
 public:
  auto getInstId(const Tensor* tensor) const -> InstId {
    AXON_ASSERT(insts_.contains(tensor));
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

  template <typename InstType, typename... Args>
  auto createTensor(Args&&... args) -> std::shared_ptr<Tensor> {
    auto inst_id = createInst<InstType>(std::forward<Args>(args)...);
    auto session = shared_from_this();
    return std::make_shared<Tensor>(session, inst_id);
  }

  template <typename InstType, typename... Args>
  auto createInst(Args&&... args) -> InstId {
    auto emit_grad = Runtime::get().shouldEmitGrad();
    auto inst_id = graph_.createOp(InstType(forward(args)...), emit_grad);
    return inst_id;
  }

  auto checkRequiresGrad(InstId inst_id) const -> bool {
    return graph_.checkRequiresGrad(inst_id);
  }

  auto performBackward(Tensor* tensor, Tensor* grad) -> void {
    auto tensor_id = getInstId(tensor);
    InstId grad_id;
    if (grad == nullptr) {
      grad_id = graph_.createOp(insts::FillLike(tensor_id, Scalar(1.0f)));
    } else {
      grad_id = getInstId(grad);
    }

    graph_.performBackward(tensor_id, grad_id);

    markAsSaved(tensor);

    evaluate();
    markForReset();
  }

  auto declareParam(std::shared_ptr<Tensor> tensor) -> void {
    AXON_DCHECK(tensor->isEvaluated());

    auto storage = tensor->storage();
    auto inst_id = graph_.declareParam(storage->shape(), storage->data_type(),
                                       tensor->requiresGrad());

    insts_[tensor.get()] = inst_id;
    parameters_.push_back(std::move(tensor));
  }

  auto merge(TraceSession& other) -> void {
    auto offset = graph_.insts().size();

    for (auto [tensor, inst_id] : other.insts_) {
      insts_[tensor] = InstId(static_cast<i32>(offset) + inst_id.value());
    }

    for (auto tensor : other.parameters_) {
      parameters_.push_back(std::move(tensor));
    }

    for (auto tensor : other.saved_) {
      saved_.push_back(tensor);
    }

    graph_.merge(other.graph_);
  }

  auto evaluate(Tensor* returned) -> void {
    AXON_ASSERT(returned != nullptr);
    AXON_ASSERT(insts_.contains(returned));

    ensureParameterGradsAreZerod();

    llvm::SmallVector<Tensor*> returned_ptrs(saved_.begin(), saved_.end());
    returned_ptrs.push_back(returned);

    Runtime::get().execute(graph_, parameters_, returned_ptrs);
  }

  auto evaluate() -> void {
    ensureParameterGradsAreZerod();

    Runtime::get().execute(graph_, parameters_, saved_);
  }

  auto markAsSaved(Tensor* tensor) -> void { saved_.push_back(tensor); }

  // Mark this session to be reset the next time a new trace starts.
  auto markForReset() -> void { should_reset_ = true; }

  // Query whether this session has been finalized by a previous
  // evaluate/backward and should not be reused for starting a new trace.
  auto shouldReset() const -> bool { return should_reset_; }

  auto graph() const -> const Graph& { return graph_; }

  auto insts() -> llvm::DenseMap<Tensor*, InstId>& { return insts_; }
  auto insts() const -> const llvm::DenseMap<Tensor*, InstId>& {
    return insts_;
  }

  auto parameters() -> llvm::SmallVector<std::shared_ptr<Tensor>>& {
    return parameters_;
  }

  auto parameters() const -> const llvm::SmallVector<std::shared_ptr<Tensor>>& {
    return parameters_;
  }

 private:
  // forward or get inst id
  auto forward(auto&& arg) -> auto {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, std::shared_ptr<Tensor>>) {
      return getInstId(arg.get());
    } else {
      return std::forward<T>(arg);
    }
  };

  // ensure that the gradients of the parameters exists and are zero'd
  auto ensureParameterGradsAreZerod() -> void {
    for (auto& p : parameters_) {
      AXON_ASSERT(p != nullptr);
      if (p->requiresGrad() && p->grad() == nullptr) {
        p->zeroGrad();
      }
    }
  }

 private:
  Graph graph_;
  llvm::DenseMap<Tensor*, InstId> insts_;
  llvm::SmallVector<std::shared_ptr<Tensor>> parameters_;
  llvm::SmallVector<Tensor*> saved_;
  bool should_reset_ = false;
};

}  // namespace axon
