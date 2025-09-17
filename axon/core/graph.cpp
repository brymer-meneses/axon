module;

#include <type_traits>

#include "axon/base/macros.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:graph;

import axon.base;

import :ids;
import :storage;
import :data_type;
import :inst;
import :shape_rules;
import :hash_rules;

namespace axon {

struct Parameter {
  Parameter() = default;
  Parameter(bool requires_grad, DataType data_type)
      : requires_grad(requires_grad), data_type(data_type) {}

  auto operator==(const Parameter& rhs) const -> bool = default;

  bool requires_grad = false;
  DataType data_type = DataType::Float32;
  InstId inst_id = InstId::None;
};

export class Graph {
 public:
  Graph() = default;

  auto operator==(const Graph& other) const -> bool = default;

  auto declareParam(llvm::ArrayRef<int64_t> shape, DataType data_type,
                    bool requires_grad) -> InstId {
    auto param_id = parameters_.emplace(Parameter(requires_grad, data_type));
    auto inst_id = insts_.emplace(insts::GetParameter(param_id));
    auto& param = parameters_.get(param_id);

    param.inst_id = inst_id;
    if (requires_grad) {
      gradients_.set(inst_id, InstId::Pending);
    }

    shapes_.set(inst_id, Shape(shape));
    data_types_.set(inst_id, data_type);
    return inst_id;
  }

  auto createConstant(Storage* constant) -> InstId {
    auto inst_id = insts_.emplace(insts::Constant{});
    constants_.set(inst_id, constant);
    data_types_.set(inst_id, constant->data_type());
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
      inferDataType(inst_id);
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
    inferDataType(inst_id);
    if (requires_grad) {
      gradients_.set(inst_id, InstId::Pending);
    }

    return inst_id;
  }

  auto merge(Graph& graph) -> void {
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

    for (auto [id, data_type] : graph.data_types_.pairs()) {
      data_types_.set(add_offset(id), data_type);
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
    llvm::hash_code hash = 0;

    for (auto [inst_id, inst] : insts_.keysAndValues()) {
      auto inst_hash = inst.visit([this](const auto& inst) {
        using InstType = std::decay_t<decltype(inst)>;
        return Hash<InstType>::hash(inst, shapes_);
      });
      auto data_type = data_types_.get(inst_id);
      if (data_type) {
        inst_hash = llvm::hash_combine(inst_hash, data_type->get());
      }
      hash = llvm::hash_combine(hash, inst_hash);
    }

    return hash;
  }

  auto reset() -> void {
    insts_.clear();
    parameters_.clear();
    constants_.clear();
    shapes_.clear();
    data_types_.clear();
    gradients_.clear();
    returned_id_ = InstId::None;
  }

  auto setReturned(InstId returned_id) -> void { returned_id_ = returned_id; }
  auto getReturnedId() const -> InstId { return returned_id_; }

  auto getDataType(InstId inst_id) const -> DataType {
    auto data_type = data_types_.get(inst_id);
    AXON_DCHECK(data_type, "No data type registered for inst");
    return data_type->get();
  }

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

  auto data_types() -> IdMap<InstId, DataType>& { return data_types_; }
  auto data_types() const -> const IdMap<InstId, DataType>& {
    return data_types_;
  }

 private:
  auto inferShape(InstId inst_id) -> void {
    auto& inst = insts_.get(inst_id);

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

  auto inferDataType(InstId inst_id) -> void {
    auto& inst = insts_.get(inst_id);

    inst.visit([&](const auto& op) {
      using InstType = std::decay_t<decltype(op)>;

      if constexpr (std::is_same_v<InstType, insts::GetParameter>) {
        auto& parameter = parameters_.get(op.param_id);
        data_types_.set(inst_id, parameter.data_type);
      } else if constexpr (std::is_same_v<InstType, insts::Constant>) {
        // Constant data type is set at creation time.
      } else if constexpr (std::is_same_v<InstType, insts::AccumulateGrad>) {
        auto value = data_types_.get(op.value_id);
        AXON_DCHECK(value, "Value data type missing");
        data_types_.set(inst_id, value->get());
      } else if constexpr (requires {
                             op.lhs_id;
                             op.rhs_id;
                           }) {
        auto lhs = data_types_.get(op.lhs_id);
        auto rhs = data_types_.get(op.rhs_id);
        AXON_DCHECK(lhs && rhs,
                    "Binary op operand data types should be available");
        AXON_DCHECK(lhs->get() == rhs->get(),
                    "Binary op operands must share the same data type");
        data_types_.set(inst_id, lhs->get());
      } else if constexpr (requires { op.operand_id; }) {
        auto operand = data_types_.get(op.operand_id);
        AXON_DCHECK(operand, "Operand data type missing");
        data_types_.set(inst_id, operand->get());
      }
    });
  }

  ValueStore<InstId, Inst> insts_;
  ValueStore<ParamId, Parameter> parameters_;

  IdMap<InstId, Storage*> constants_;
  IdMap<InstId, Shape> shapes_;
  IdMap<InstId, DataType> data_types_;

  IdStore<InstId, InstId> gradients_;

  InstId returned_id_;
};

}  // namespace axon

export template <>
struct std::hash<axon::Graph> {
  auto operator()(const axon::Graph& graph) const -> size_t {
    return graph.hash();
  }
};
