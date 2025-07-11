module;

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

export module axon.core:inst;

import :ids;
import :inst_kinds;

export namespace axon {

template <typename... Args>
struct match : Args... {
  using Args::operator()...;
};

class Inst {
 public:
  using Value =
      std::variant<insts::MatMul, insts::Add, insts::Mul, insts::Transpose,
                   insts::GetFunctionArgument, insts::GetCachedValue,
                   insts::Write, insts::Constant>;

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

  auto index() const -> int32_t { return value_.index(); }

  auto is_expression() const -> bool {
    static constexpr auto visitor = match{
        [](const insts::Mul&) { return true; },
        [](const insts::MatMul&) { return true; },
        [](const insts::Add&) { return true; },
        [](const insts::Constant&) { return true; },
        [](const auto&) { return false; },
    };

    return std::visit(visitor, value_);
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
  Value value_;
};

}  // namespace axon
