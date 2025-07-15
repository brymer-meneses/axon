module;

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "axon/base/dcheck.h"

export module axon.base:index_base;

export namespace axon {

template <typename T>
struct IndexBase {
  constexpr explicit IndexBase(int32_t value) : value_(value) {}
  constexpr auto has_value() const -> bool { return value_ >= 0; }
  constexpr auto value() -> int32_t {
    AXON_DCHECK(has_value(),
                "Tried to call value() on an index that has a value of {}.",
                value_);
    return value_;
  }

  friend auto operator<=>(const T lhs, const T rhs) -> auto {
    return lhs.value_ <=> rhs.value_;
  }

  friend auto operator==(const T lhs, const T rhs) -> bool {
    return lhs.value_ == rhs.value_;
  }

 private:
  int32_t value_;
};

template <typename IndexType>
concept Index = std::is_base_of_v<IndexBase<IndexType>, IndexType>;

}  // namespace axon
