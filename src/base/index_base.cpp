module;

#include <cassert>
#include <cstdint>

export module axon.base.index;

namespace axon {

export template <typename T>
struct IndexBase {
  constexpr explicit IndexBase(int32_t value) : value_(value) {}
  constexpr auto has_value() const -> bool { return value_ >= 0; }
  constexpr auto value() -> int32_t {
    assert(has_value());
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

}  // namespace axon
