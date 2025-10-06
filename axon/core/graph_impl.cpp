module;

#include <type_traits>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

module axon.core:graph_impl;

import axon.base;

import :backward_rules;
import :shape_rules;
import :data_type;
import :hash_rules;
import :ids;
import :storage;
import :graph;

namespace axon {

auto Graph::performBackward(InstId output_id, InstId grad_id) -> void {
  AXON_ASSERT(output_id.isValid(), "`output_id` has no value.");
  AXON_ASSERT(grad_id.isValid());

  llvm::SmallVector<Dependency> work_list;
  work_list.emplace_back(output_id, grad_id);

  BackwardContext ctx{*this};

  while (not work_list.empty()) {
    auto dep = work_list.pop_back_val();

    ctx.accumulateGrad(dep);

    // This should take by copy since `graph.insts` will grow.
    auto inst = insts_.get(dep.inst_id);

    // Provide current inst id to backward rules that may reference it.
    ctx.setCurrentInstId(dep.inst_id);

    inst.visit([&, dep](const auto& op) {
      using InstType = std::decay_t<decltype(op)>;
      if constexpr (HasBackwardRule<InstType>) {
        auto new_deps = BackwardRule<InstType>::apply(op, dep.grad_id, ctx);
        for (auto dep : new_deps) {
          work_list.emplace_back(dep);
        }
      }
    });
  }

  for (auto& param : parameters_.values()) {
    if (!param.requires_grad) {
      continue;
    }

    auto grad_id = gradients_.get(param.inst_id);
    if (grad_id.isValid()) {
      createOp(insts::AccumulateGrad(param.inst_id, grad_id),
               /*emit_grad=*/false);
    }
  }
}

auto Graph::declareParam(llvm::ArrayRef<int64_t> shape, DataType data_type,
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

auto Graph::createConstant(Storage* constant) -> InstId {
  auto inst_id = insts_.emplace(insts::Constant{});
  constants_.set(inst_id, constant);
  data_types_.set(inst_id, constant->data_type());
  return inst_id;
}

auto Graph::createOp(Inst&& inst, bool emit_grad) -> InstId {
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
    } else if constexpr (InstType::traits.num_inputs == 2) {
      return checkRequiresGrad(op.lhs_id) || checkRequiresGrad(op.rhs_id);
    } else if constexpr (InstType::traits.num_inputs == 1) {
      return checkRequiresGrad(op.input_id);
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

auto Graph::merge(Graph& graph) -> void {
  AXON_ASSERT(!graph.returned_id_.isValid());

  auto param_offset = static_cast<i32>(parameters_.size());
  auto inst_offset = static_cast<i32>(insts_.size());

  for (auto& param : graph.parameters().values()) {
    param.inst_id = param.inst_id.add_offset(inst_offset);
    parameters_.add(std::move(param));
  }

  for (auto& [key, value] : graph.gradients_.pairs()) {
    gradients_.set(key.add_offset(inst_offset), value.add_offset(inst_offset));
  }

  for (auto [inst_id, shape] : graph.shapes_.pairs()) {
    shapes_.set(inst_id.add_offset(inst_offset), std::move(shape));
  }

  for (auto [inst_id, constant] : graph.constants_.pairs()) {
    constants_.set(inst_id.add_offset(inst_offset), std::move(constant));
  }

  for (auto [inst_id, data_type] : graph.data_types_.pairs()) {
    data_types_.set(inst_id.add_offset(inst_offset), data_type);
  }

  auto add_offset_to_inst = [inst_offset, param_offset](auto& op) -> void {
    using InstType = std::decay_t<decltype(op)>;
    if constexpr (llvm::is_one_of<InstType, insts::AccumulateGrad,
                                  insts::AccumulateData>()) {
      op.sink_id = op.sink_id.add_offset(inst_offset);
      op.source_id = op.source_id.add_offset(inst_offset);
    } else if constexpr (std::is_same_v<InstType, insts::GetParameter>) {
      op.param_id = op.param_id.add_offset(param_offset);
    } else if constexpr (InstType::traits.num_inputs == 2) {
      op.lhs_id = op.lhs_id.add_offset(inst_offset);
      op.rhs_id = op.rhs_id.add_offset(inst_offset);
    } else if constexpr (InstType::traits.num_inputs == 1) {
      op.input_id = op.input_id.add_offset(inst_offset);
    } else if constexpr (InstType::traits.num_inputs == 0) {
    } else {
      static_assert(false, "Unhandled inst");
    }
  };

  for (Inst& inst : graph.insts_.values()) {
    inst.visit(add_offset_to_inst);
    insts_.emplace(std::move(inst));
  }
}

auto Graph::hash() const -> u64 {
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

auto Graph::reset() -> void {
  insts_.clear();
  parameters_.clear();
  constants_.clear();
  shapes_.clear();
  data_types_.clear();
  gradients_.clear();
  returned_id_ = InstId::None;
}

auto Graph::inferShape(InstId inst_id) -> void {
  auto& inst = insts_.get(inst_id);

  inst.visit([&](const auto& op) {
    using InstType = std::decay_t<decltype(op)>;

    if constexpr (InstType::traits.shape_rule == ShapeInfo::SameAsInputs) {
      if constexpr (InstType::traits.num_inputs == 2) {
        auto lhs_id = op.lhs_id;
        Shape shape = *shapes_.get(lhs_id);
        shapes_.set(inst_id, std::move(shape));
      } else if constexpr (InstType::traits.num_inputs == 1) {
        Shape shape = *shapes_.get(op.input_id);
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

auto Graph::inferDataType(InstId inst_id) -> void {
  auto& inst = insts_.get(inst_id);

  inst.visit([&](const auto& op) {
    using InstType = std::decay_t<decltype(op)>;

    if constexpr (std::is_same_v<InstType, insts::GetParameter>) {
      auto& parameter = parameters_.get(op.param_id);
      data_types_.set(inst_id, parameter.data_type);
    } else if constexpr (std::is_same_v<InstType, insts::Constant>) {
      // Constant data type is set at creation time.
    } else if constexpr (std::is_same_v<InstType, insts::AccumulateGrad>) {
      auto value = data_types_.get(op.source_id);
      AXON_ASSERT(value, "Value data type missing");
      data_types_.set(inst_id, value->get());
    } else if constexpr (std::is_same_v<InstType, insts::ArgMax>) {
      data_types_.set(inst_id, DataType::Int64);
    } else if constexpr (requires {
                           op.lhs_id;
                           op.rhs_id;
                         }) {
      auto lhs = data_types_.get(op.lhs_id);
      auto rhs = data_types_.get(op.rhs_id);
      AXON_ASSERT(lhs && rhs,
                  "Binary op operand data types should be available");
      AXON_ASSERT(lhs->get() == rhs->get(),
                  "Binary op operands must share the same data type");
      data_types_.set(inst_id, lhs->get());
    } else if constexpr (requires { op.input_id; }) {
      auto operand = data_types_.get(op.input_id);
      AXON_ASSERT(operand, "Operand data type missing");
      data_types_.set(inst_id, operand->get());
    }
  });
}

}  // namespace axon
