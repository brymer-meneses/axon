module;

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

export module axon.core.inst;

import axon.core.ids;

namespace axon {

namespace insts {

export struct Copy {
  static constexpr bool Differentiable = false;

  InstId inst_id;
  IntermediaryValueId value_id;
};

export struct GetInitialGradient {
  static constexpr bool Differentiable = false;
};

export struct GetParameter {
  static constexpr bool Differentiable = false;

  ParamId param_id;
};

export struct GetIntermediaryValue {
  static constexpr bool Differentiable = false;
  IntermediaryValueId value_id;
};

export struct Add {
  static constexpr bool Differentiable = true;

  InstId lhs_id;
  InstId rhs_id;
};

export struct Mul {
  static constexpr bool Differentiable = true;

  InstId lhs_id;
  InstId rhs_id;
};

export struct MatMul {
  static constexpr bool Differentiable = false;

  InstId lhs_id;
  InstId rhs_id;
};

}  // namespace insts

export template <typename... Args>
struct match : Args... {
  using Args::operator()...;
};

export class Inst {
 public:
  using Value = std::variant<insts::Add, insts::Mul, insts::MatMul,
                             insts::GetParameter, insts::GetIntermediaryValue,
                             insts::Copy, insts::GetInitialGradient>;

  template <typename InstType>
    requires std::is_convertible_v<InstType, Value>
  Inst(InstType&& inst) : value_(inst) {}

  template <typename VisitorType>
  auto visit(VisitorType&& visitor) -> auto {
    return std::visit(visitor, value_);
  }

  template <typename InstType>
    requires std::is_convertible_v<InstType, Value>
  auto try_get_as() const -> std::optional<InstType> {
    if (auto* value = std::get_if<InstType>(&value_)) {
      return std::make_optional<InstType>(*value);
    }
    return std::nullopt;
  }

  // Returns the parents of an instruction.
  auto parents() const -> llvm::SmallVector<InstId> {
    static constexpr auto visitor = match{
        [](const insts::Add& inst) {
          return llvm::SmallVector<InstId>{inst.lhs_id, inst.rhs_id};
        },
        [](const insts::Mul& inst) {
          return llvm::SmallVector<InstId>{inst.lhs_id, inst.rhs_id};
        },
        [](const insts::MatMul& inst) -> llvm::SmallVector<InstId> {
          return llvm::SmallVector<InstId>{inst.lhs_id, inst.rhs_id};
        },
        [](const auto&) { return llvm::SmallVector<InstId>{}; },
    };
    return std::visit(visitor, value_);
  }

 private:
  Value value_;
};

}  // namespace axon
