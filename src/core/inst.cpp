module;

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

export module axon.core.inst;

import axon.core.ids;

namespace axon {

namespace insts {

export struct Declare {};

export struct DeclareParameter {
  ParamId param_id;
};

export struct Add {
  InstId lhs_id;
  InstId rhs_id;
};

export struct Mul {
  InstId lhs_id;
  InstId rhs_id;
};

export struct MatMul {
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
  using Value = std::variant<insts::Declare, insts::Add, insts::Mul,
                             insts::MatMul, insts::DeclareParameter>;

  template <typename InstType>
    requires std::is_convertible_v<InstType, Value>
  Inst(InstType&& inst) : value_(inst) {}

  template <typename VisitorType>
    requires std::is_invocable_v<VisitorType, Value>
  auto visit(VisitorType&& visitor) -> auto {
    return visitor(value_);
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
