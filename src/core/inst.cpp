module;

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

export module axon.core.inst;

import axon.core.ids;

export namespace axon {

namespace insts {

struct GetParameter {
  static constexpr bool Differentiable = false;

  ParamId param_id;
};

struct GetCachedValue {
  static constexpr bool Differentiable = false;
  CachedValueId value_id;
};

struct Add {
  static constexpr bool Differentiable = true;

  InstId lhs_id;
  InstId rhs_id;
};

struct Mul {
  static constexpr bool Differentiable = true;

  InstId lhs_id;
  InstId rhs_id;
};

struct MatMul {
  static constexpr bool Differentiable = false;

  InstId lhs_id;
  InstId rhs_id;
};

struct Copy {
  static constexpr bool Differentiable = false;

  InstId inst_id;
  CachedValueId value_id;
};

}  // namespace insts

template <typename... Args>
struct match : Args... {
  using Args::operator()...;
};

class Inst {
 public:
  using Value =
      std::variant<insts::Add, insts::Mul, insts::MatMul, insts::GetParameter,
                   insts::GetCachedValue, insts::Copy>;

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
