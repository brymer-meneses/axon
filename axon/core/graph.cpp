module;

#include "axon/base/macros.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:graph;

import axon.base;

import :ids;
import :storage;
import :inst;
import :shape_rules;
import :hash_rules;

namespace axon {

struct Parameter {
  Parameter(bool requires_grad) : requires_grad(requires_grad) {}

  auto operator==(const Parameter& rhs) const -> bool = default;

  bool requires_grad;
  InstId inst_id = InstId::None;
};

export class Graph {
 public:
  Graph() = default;

  auto operator==(const Graph& other) const -> bool = default;

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

  auto createConstant(Storage* constant) -> InstId {
    auto inst_id = insts_.emplace(insts::Constant{});
    constants_.set(inst_id, constant);
    return inst_id;
  }

  auto getShape(InstId inst_id) const -> ShapeRef {
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

  auto absorb(Graph& graph) -> void {
    AXON_DCHECK(!graph.returned_id_.isValid());

    auto add_offset = [offset =
                           insts_.size()](InstId foreign_inst_id) -> InstId {
      return InstId(static_cast<i32>(offset) + foreign_inst_id.value());
    };

    for (auto& param : graph.parameters().values()) {
      param.inst_id = add_offset(param.inst_id);
      parameters_.add(std::move(param));
    }

    for (auto& [key, value] : graph.gradients_.pairs()) {
      gradients_.set(add_offset(key), add_offset(value));
    }

    for (auto [id, shape] : graph.shapes_.pairs()) {
      shapes_.set(add_offset(id), std::move(shape));
    }

    for (auto [id, constant] : graph.constants_.pairs()) {
      constants_.set(add_offset(id), std::move(constant));
    }

    auto param_size = static_cast<i32>(parameters_.size());

    auto add_offset_to_inst = [add_offset, param_size](auto& op) -> void {
      using InstType = std::decay_t<decltype(op)>;
      if constexpr (std::is_same_v<InstType, insts::AccumulateGrad>) {
        op.inst_id = add_offset(op.value_id);
        op.value_id = add_offset(op.value_id);
      } else if constexpr (std::is_same_v<InstType, insts::GetParameter>) {
        op.param_id = ParamId(param_size + op.param_id.value());
      } else if constexpr (InstType::traits.num_operands == 2) {
        op.lhs_id = add_offset(op.lhs_id);
        op.rhs_id = add_offset(op.rhs_id);
      } else if constexpr (InstType::traits.num_operands == 1) {
        op.operand_id = add_offset(op.operand_id);
      } else {
        // static_assert(false, "Unhandled inst");
      }
    };

    for (Inst& inst : graph.insts_.values()) {
      inst.visit(add_offset_to_inst);
      insts_.emplace(std::move(inst));
    }
  }

  auto hash() const -> u64 {
    llvm::hash_code hash;
    for (auto& inst : insts_.values()) {
      auto inst_hash = inst.visit([this](const auto& inst) {
        using InstType = std::decay_t<decltype(inst)>;
        return Hash<InstType>::hash(inst, shapes_);
      });
      hash = llvm::hash_combine(hash, inst_hash);
    }
    return hash;
  }

  auto setReturned(InstId returned_id) -> void { returned_id_ = returned_id; }

  auto getReturnedId() const -> InstId { return returned_id_; }

  auto gradients() -> IdStore<InstId, InstId>& { return gradients_; }
  auto gradients() const -> const IdStore<InstId, InstId>& {
    return gradients_;
  }

  auto insts() -> ValueStore<InstId, Inst>& { return insts_; }
  auto insts() const -> const ValueStore<InstId, Inst>& { return insts_; }

  auto constants() -> IdMap<InstId, Storage*>& { return constants_; }
  auto constants() const -> const IdMap<InstId, Storage*>& {
    return constants_;
  }

  auto parameters() -> ValueStore<ParamId, Parameter>& { return parameters_; }
  auto parameters() const -> const ValueStore<ParamId, Parameter>& {
    return parameters_;
  }

  auto shapes() -> IdMap<InstId, Shape>& { return shapes_; }
  auto shapes() const -> const IdMap<InstId, Shape>& { return shapes_; }

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

  IdMap<InstId, Storage*> constants_;
  IdMap<InstId, Shape> shapes_;

  IdStore<InstId, InstId> gradients_;

  InstId returned_id_;
};

}  // namespace axon

export template <>
struct llvm::DenseMapInfo<axon::Graph> {
  static inline axon::Graph* empty_key = reinterpret_cast<axon::Graph*>(-1);
  static inline axon::Graph* tombstone_key = reinterpret_cast<axon::Graph*>(-2);

  static inline auto getEmptyKey() -> axon::Graph { return *empty_key; }

  static inline auto getTombstoneKey() -> axon::Graph { return *tombstone_key; }

  static auto getHashValue(const axon::Graph& graph) -> unsigned {
    return graph.hash();
  }

  static auto isEqual(const axon::Graph& lhs, const axon::Graph& rhs) -> bool {
    if (&lhs == empty_key || &lhs == tombstone_key || &rhs == empty_key ||
        &rhs == tombstone_key) {
      return &lhs == &rhs;
    }

    // Try cheap paths first.

    if (lhs.getReturnedId() != rhs.getReturnedId()) {
      return false;
    }
    if (lhs.gradients().size() != rhs.gradients().size()) {
      return false;
    }
    if (lhs.constants().size() != rhs.constants().size()) {
      return false;
    }
    if (lhs.parameters().size() != rhs.parameters().size()) {
      return false;
    }
    if (lhs.insts().size() != rhs.insts().size()) {
      return false;
    }

    return lhs == rhs;
  }
};
