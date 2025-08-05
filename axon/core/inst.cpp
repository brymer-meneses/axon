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
  auto is() const -> bool {
    return std::get_if<InstType>(&value_) != nullptr;
  }

  template <typename InstType>
    requires std::is_convertible_v<InstType, InstInternalType>
  auto tryGetAs() const -> std::optional<InstType> {
    if (auto* value = std::get_if<InstType>(&value_)) {
      return std::make_optional<InstType>(*value);
    }
    return std::nullopt;
  }

  template <typename InstType>
    requires std::is_convertible_v<InstType, InstInternalType>
  auto getAs() const -> InstType {
    auto* value = std::get_if<InstType>(&value_);
    AXON_DCHECK(value != nullptr, "Invalid type.");
    return *value;
  }

  auto index() const -> int32_t { return value_.index(); }

  auto isExpression() const -> bool {
    return std::visit(
        [](const auto& op) {
          using InstType = std::decay_t<decltype(op)>;
          return IsExpressionInst<InstType>;
        },
        value_);
  }

 private:
  InstInternalType value_;
};

}  // namespace axon
