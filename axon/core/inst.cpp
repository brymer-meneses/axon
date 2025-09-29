module;

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

export module axon.core:inst;

import :ids;
import :inst_kinds;

import axon.base;

namespace axon {

template <typename VariantType, typename T, std::size_t index = 0>
consteval auto getVariantIndex() -> size_t {
  static_assert(std::variant_size_v<VariantType> > index,
                "Type not found in variant");
  if constexpr (index == std::variant_size_v<VariantType>) {
    return index;
  } else if constexpr (std::is_same_v<
                           std::variant_alternative_t<index, VariantType>, T>) {
    return index;
  } else {
    return getVariantIndex<VariantType, T, index + 1>();
  }
}

template <typename T>
concept InstConvertible = std::is_convertible_v<T, InstInternalType>;

template <typename VisitorType>
concept InstVisitor = requires(VisitorType visitor, InstInternalType inst) {
  std::visit(visitor, inst);
};

export class Inst {
 public:
  template <InstConvertible InstType>
  Inst(InstType&& inst) : value_(inst) {}

  auto operator==(const Inst& other) const -> bool {
    if (value_.index() != other.index()) {
      return false;
    }
    return value_ == other.value_;
  }

  template <InstVisitor VisitorType>
  auto visit(VisitorType&& visitor) const -> auto {
    return std::visit(visitor, value_);
  }

  template <InstVisitor VisitorType>
  auto visit(VisitorType&& visitor) -> auto {
    return std::visit(visitor, value_);
  }

  template <InstConvertible InstType>
  auto is() const -> bool {
    return std::get_if<InstType>(&value_) != nullptr;
  }

  template <InstConvertible InstType>
  auto tryGetAs() const -> std::optional<InstType> {
    if (auto* value = std::get_if<InstType>(&value_)) {
      return std::make_optional<InstType>(*value);
    }
    return std::nullopt;
  }

  template <InstConvertible InstType>
  auto getAs() const -> InstType {
    auto* value = std::get_if<InstType>(&value_);
    AXON_ASSERT(value != nullptr, "Invalid type.");
    return *value;
  }

  auto index() const -> size_t { return value_.index(); }

  template <InstConvertible T>
  static consteval auto tag() -> size_t {
    return getVariantIndex<InstInternalType, T>();
  }

 private:
  InstInternalType value_;
};

}  // namespace axon
