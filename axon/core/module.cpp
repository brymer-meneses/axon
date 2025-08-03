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

  auto check_requires_grad(InstId inst_id) const -> bool {
    if (auto get_input =
            forward_.insts().get(inst_id).try_get_as<insts::GetInput>()) {
      return forward_.inputs().get(get_input->input_id).requires_grad;
    }
    return gradients_.contains(inst_id);
  }

  auto declare_input(llvm::SmallVector<int64_t> shape, bool requires_grad)
      -> Tensor {
    auto inst_id = forward_.declare_input(shape, requires_grad);
    return {inst_id};
  }

  auto create_return(Tensor tensor) -> void {
    forward_.emit(axon::insts::Return(tensor.inst_id));
  }

  auto emit(Inst inst) -> Tensor {
    auto inst_id = forward_.emit(inst);
    return {inst_id};
  }

  auto is_finalized() const -> bool { return backward_.insts().size() > 0; }

  auto gradients() -> auto& { return gradients_; }
  auto gradients() const -> const auto& { return gradients_; }

  auto forward() -> auto& { return forward_; }
  auto forward() const -> const auto& { return forward_; }

  auto backward() -> auto& { return backward_; }
  auto backward() const -> const auto& { return backward_; }

 private:
  Graph forward_;
  Graph backward_;
  IdStore<InstId, InstId> gradients_;
  std::shared_ptr<Context> context_;
};

}  // namespace axon
