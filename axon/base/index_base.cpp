module;

#include <limits>

#include "llvm/ADT/DenseMapInfo.h"

export module axon.base:index_base;

export namespace axon {

template <typename T>
struct IndexBase {
  constexpr explicit IndexBase(int32_t value) : value_(value) {}
  constexpr explicit IndexBase() = default;

  constexpr auto isValid() const -> bool { return value_ >= 0; }
  constexpr auto value() const -> int32_t { return value_; }

  constexpr operator bool() const { return isValid(); }

  friend auto operator<=>(const T lhs, const T rhs) -> auto {
    return lhs.value_ <=> rhs.value_;
  }

  friend auto operator==(const T lhs, const T rhs) -> bool {
    return lhs.value_ == rhs.value_;
  }

  static const T None;

 private:
  int32_t value_ = -1;
};

template <typename IndexType>
concept Index = std::is_base_of_v<IndexBase<IndexType>, IndexType>;

template <typename T>
constexpr T IndexBase<T>::None = T(-1);

}  // namespace axon

namespace llvm {

export template <axon::Index T>
struct DenseMapInfo<T> {
  static constexpr auto sentinel = std::numeric_limits<int32_t>::min();

  static inline auto getEmptyKey() -> T { return T(sentinel); }

  static inline auto getTombstoneKey() -> T { return T(sentinel + 1); }

  static auto getHashValue(const T& value) -> unsigned {
    return static_cast<unsigned>(value.value() * 37UL);
  }

  static auto isEqual(const T& lhs, const T& rhs) -> bool {
    return lhs.value() == rhs.value();
  }
};

}  // namespace llvm
