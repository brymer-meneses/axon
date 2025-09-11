module;

#include "axon/base/macros.h"
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

  auto createOp(Inst&& inst, bool emit_grad = true) -> InstId {
    if (not emit_grad) {
      auto inst_id = insts_.emplace(std::move(inst));
      inferShape(inst_id);
      return inst_id;
    }

    auto check_requires_grad = [this](const auto& op) -> bool {
      using InstType = std::decay_t<decltype(op)>;
      if constexpr (!InstType::traits.differentiable) {
        return false;
      } else if constexpr (InstType::traits.num_operands == 2) {
        return checkRequiresGrad(op.lhs_id) || checkRequiresGrad(op.rhs_id);
      } else if constexpr (InstType::traits.num_operands == 1) {
        return checkRequiresGrad(op.operand_id);
      }

      return false;
    };

    auto requires_grad = inst.visit(check_requires_grad);
    auto inst_id = insts_.emplace(std::move(inst));
    inferShape(inst_id);
    if (requires_grad) {
      gradients_.set(inst_id, InstId::Pending);
    }

    return inst_id;
  }

  auto setReturned(InstId returned_id) -> void { returned_id_ = returned_id; }

  auto getReturnedId() const -> InstId { return returned_id_; }

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

    inst.visit([&](const auto& op) {
      using InstType = std::decay_t<decltype(op)>;

      if constexpr (InstType::traits.shape_rule == ShapeInfo::SameAsOperands) {
        if constexpr (InstType::traits.num_operands == 2) {
          auto [lhs_id, rhs_id] = op;
          Shape shape = *shapes_.get(lhs_id);
          shapes_.set(inst_id, std::move(shape));
        } else if constexpr (InstType::traits.num_operands == 1) {
          Shape shape = *shapes_.get(op.operand_id);
          shapes_.set(inst_id, std::move(shape));
        }
      } else if constexpr (InstType::traits.shape_rule == ShapeInfo::Custom) {
        auto shape = InferShapeRule<InstType>::apply(op, shapes_);
        shapes_.set(inst_id, std::move(shape));
      } else if constexpr (InstType::traits.shape_rule == ShapeInfo::None) {
        // do nothing
      } else {
        static_assert(false, "Unhandled operation");
      }
    });
  }

  ValueStore<InstId, Inst> insts_;
  ValueStore<ParamId, Parameter> parameters_;

  IdMap<InstId, Storage> constants_;
  IdMap<InstId, Shape> shapes_;

  IdStore<InstId, InstId> gradients_;

  InstId returned_id_;
};

}  // namespace axon
