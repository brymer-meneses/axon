module;

#include "axon/base/dcheck.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:graph;

import axon.base;

import :ids;
import :storage;
import :inst;
import :shape_rules;

namespace axon {

struct Parameter {
  Parameter(bool requires_grad) : requires_grad(requires_grad) {}

  bool requires_grad;
  InstId inst_id = InstId::None;
};

export class Graph {
 public:
  Graph() = default;

  Graph(const Graph&) = delete;
  auto operator=(const Graph&) -> Graph& = delete;

  auto declareParam(llvm::ArrayRef<int64_t> shape, bool requires_grad)
      -> InstId {
    auto param_id = parameters_.emplace(requires_grad);
    auto inst_id = insts_.emplace(insts::GetParameter(param_id));
    auto& param = parameters_.get(param_id);

    param.inst_id = inst_id;
    if (requires_grad) {
      gradients_.set(inst_id, InstId::Pending);
    }

    shapes_.set(inst_id, Shape(shape));
    return inst_id;
  }

  auto createConstant(Storage&& constant) -> InstId {
    auto inst_id = insts_.emplace(insts::Constant{});
    constants_.set(inst_id, std::move(constant));
    return inst_id;
  }

  auto getShape(InstId inst_id) -> ShapeRef {
    if (auto shape = shapes_.get(inst_id)) {
      return shape->get();
    };
    return {};
  }

  auto checkRequiresGrad(InstId inst_id) const -> bool {
    return gradients_.containsKey(inst_id);
  }

  auto createOp(Inst inst, bool emit_grad = true) -> InstId {
    auto inst_id = insts_.emplace(inst);
    if (emit_grad) {
      auto parents = inst.parents();
      auto requires_grad = std::ranges::any_of(
          parents,
          [this](InstId parent_id) { return checkRequiresGrad(parent_id); });

      if (requires_grad) {
        gradients_.set(inst_id, InstId::Pending);
      }
    }

    inferShape(inst_id);
    return inst_id;
  }

  auto gradients() -> auto& { return gradients_; }
  auto gradients() const -> const auto& { return gradients_; }

  auto insts() -> auto& { return insts_; }
  auto insts() const -> const auto& { return insts_; }

  auto constants() -> auto& { return constants_; }
  auto constants() const -> const auto& { return constants_; }

  auto parameters() -> auto& { return parameters_; }
  auto parameters() const -> const auto& { return parameters_; }

  auto shapes() -> auto& { return shapes_; }
  auto shapes() const -> const auto& { return shapes_; }

 private:
  auto inferShape(InstId inst_id) -> void {
    auto inst = insts_.get(inst_id);

    inst.visit([&](const auto op) {
      using InstType = std::decay_t<decltype(op)>;

      if constexpr (HasInferShapeRule<InstType>) {
        auto shape = InferShapeRule<InstType>::apply(op, shapes_);
        shapes_.set(inst_id, std::move(shape));
        return;
      }

      if constexpr (llvm::is_one_of<insts::Mul, insts::Add, insts::MatMul>()) {
        auto [lhs_id, rhs_id] = op;
        auto shape = *shapes_.get(lhs_id);
        shapes_.set(inst_id, std::move(shape));
        return;
      }

      if constexpr (llvm::is_one_of<insts::Transpose, insts::Squeeze,
                                    insts::Unsqueeze, insts::Broadcast>()) {
        auto shape = *shapes_.get(op.operand_id);
        shapes_.set(inst_id, std::move(shape));
        return;
      }
    });
  }

  ValueStore<InstId, Inst> insts_;
  ValueStore<ParamId, Parameter> parameters_;

  IdMap<InstId, Storage> constants_;
  IdMap<InstId, Shape> shapes_;

  IdStore<InstId, InstId> gradients_;
};

}  // namespace axon
