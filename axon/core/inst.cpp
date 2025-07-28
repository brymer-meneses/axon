module;

#include <optional>

#include "axon/base/dcheck.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

export module axon.core:inst;

import :ids;
import :inst_kinds;

import axon.base;

export namespace axon {

class Inst {
 public:
  template <typename InstType>
    requires std::is_convertible_v<InstType, InstInternalType>
  Inst(InstType&& inst) : value_(inst) {}

  template <typename VisitorType>
  auto visit(VisitorType&& visitor) const -> auto {
    return std::visit(visitor, value_);
  }

  template <typename InstType>
    requires std::is_convertible_v<InstType, InstInternalType>
  auto try_get_as() const -> std::optional<InstType> {
    if (auto* value = std::get_if<InstType>(&value_)) {
      return std::make_optional<InstType>(*value);
    }
    return std::nullopt;
  }

  template <typename InstType>
    requires std::is_convertible_v<InstType, InstInternalType>
  auto get_as() const -> InstType {
    auto* value = std::get_if<InstType>(&value_);
    AXON_DCHECK(value != nullptr, "Invalid type.");
    return *value;
  }

  auto index() const -> int32_t { return value_.index(); }

  auto is_expression() const -> bool {
    return std::visit(
        [](const auto& op) {
          using InstType = std::decay_t<decltype(op)>;
          return IsExpressionInst<InstType>;
        },
        value_);
  }

  // Returns the parents of an instruction.
  auto parents() const -> llvm::SmallVector<InstId> {
    static constexpr auto visitor = match{
        [](const insts::Add& inst) -> llvm::SmallVector<InstId> {
          return {inst.lhs_id, inst.rhs_id};
        },
        [](const insts::Mul& inst) -> llvm::SmallVector<InstId> {
          return {inst.lhs_id, inst.rhs_id};
        },
        [](const insts::MatMul& inst) -> llvm::SmallVector<InstId> {
          return {inst.lhs_id, inst.rhs_id};
        },
        [](const auto&) { return llvm::SmallVector<InstId>{}; },
    };
    return std::visit(visitor, value_);
  }

 private:
  InstInternalType value_;
};

}  // namespace axon
