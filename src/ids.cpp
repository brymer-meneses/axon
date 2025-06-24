module;

#include <cassert>
#include <cstdint>

export module axon.ids;

namespace axon {

export template <typename T>
struct IndexBase {
  constexpr IndexBase(int32_t value) : value_(value) {}
  constexpr auto has_value() const -> bool { return value_ >= 0; }
  constexpr auto value() -> int32_t {
    assert(has_value());

    return value_;
  }

  friend auto operator==(const T lhs, const T rhs) -> bool {
    return lhs.value_ == rhs.value_;
  }

 private:
  int32_t value_;
};

export struct InstId : IndexBase<InstId> {
  using IndexBase::IndexBase;

  static const InstId Invalid;
};

inline constexpr auto InstId::Invalid = InstId(-1);

export struct DataId : IndexBase<DataId> {
  using IndexBase::IndexBase;

  static const DataId Invalid;
  static const DataId Pending;
  static const DataId Deallocated;
};

inline constexpr auto DataId::Invalid = DataId(-1);
inline constexpr auto DataId::Pending = DataId(-2);
inline constexpr auto DataId::Deallocated = DataId(-3);

}  // namespace axon
