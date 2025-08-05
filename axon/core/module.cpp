module;

#include "llvm/ADT/SmallVector.h"

export module axon.core:mod;

import std;
import :graph;

export namespace axon {

struct Tensor {
  InstId inst_id;

  constexpr operator InstId() { return inst_id; }
};

class Module {
 public:
  Module(std::shared_ptr<Context> context) : context_(std::move(context)) {}

  auto checkRequiresGrad(InstId inst_id) const -> bool {
    if (auto get_input =
            forward_.insts().get(inst_id).tryGetAs<insts::GetInput>()) {
      return forward_.inputs().get(get_input->input_id).requires_grad;
    }
    return gradients_.contains(inst_id);
  }

  auto declareInput(llvm::SmallVector<int64_t> shape, bool requires_grad)
      -> Tensor {
    auto inst_id = forward_.declareInput(shape, requires_grad);
    return {inst_id};
  }

  auto createReturn(Tensor tensor) -> void {
    forward_.setOutput(tensor.inst_id);
  }

  auto emit(Inst inst) -> Tensor {
    auto inst_id = forward_.emit(inst);
    return {inst_id};
  }

  auto isFinalized() const -> bool { return backward_.insts().size() > 0; }

  auto forward() -> auto& { return forward_; }
  auto forward() const -> const auto& { return forward_; }

  auto gradients() -> auto& { return gradients_; }
  auto gradients() const -> const auto& { return gradients_; }

  auto backward() -> auto& { return backward_; }
  auto backward() const -> const auto& { return backward_; }

 private:
  Graph forward_;
  Graph backward_;
  IdStore<InstId, InstId> gradients_;
  std::shared_ptr<Context> context_;
};

}  // namespace axon
