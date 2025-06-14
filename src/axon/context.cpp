module;

#include <print>
#include <string>
#include <vector>

export module axon.context;

import axon.op;
import axon.tensor;
import axon.common;

namespace axon {

export struct Inst {
  Operation op;
  TensorId result;
};

export struct InstId : IdBase<InstId, Inst> {
  using IdBase::IdBase;
};

export class Context {
 public:
  Context(const std::string& module_name) : module_name_(module_name) {}

  auto declare_tensor(bool requires_grad, std::vector<int16_t> shape)
      -> TensorId {
    return tensors_.append({
        .requires_grad = requires_grad,
        .is_optimizable = false,
        .shape = std::move(shape),
    });
  }

  auto record_batch_matmul(TensorId tid1, TensorId tid2) -> void {
    const auto& t1 = tensors_[tid1];
    const auto& t2 = tensors_[tid2];

    if (t1.shape[0] != t2.shape[0]) {
      throw std::runtime_error("Invalid batch parameters");
    }

    const auto requires_grad = t1.requires_grad or t2.requires_grad;

    // (N, A, B) @ (N, B, C) -> (N, A, C)
    const auto result =
        declare_tensor(requires_grad, {t1.shape[0], t1.shape[1], t2.shape[2]});

    insts_.append({.op = BatchMatMul(tid1, tid2), .result = result});
  }

  auto compile() -> void {}

 private:
  const std::string& module_name_;
  BasicStorage<TensorId> tensors_;
  BasicStorage<InstId> insts_;
};

}  // namespace axon
