module;

#include <cassert>
#include <cstdint>

export module axon.ids;

namespace axon {

export class IndexBase {
 public:
  constexpr auto is_valid() const -> bool { return value_ != -1; }
  constexpr auto operator*() const -> int32_t {
    assert(value_ != -1);
    return value_;
  }
  constexpr explicit IndexBase(int32_t value) : value_(value) {
    assert(value != -1);
  }

 private:
  int32_t value_;
};

export struct TensorId : IndexBase {
  using IndexBase::IndexBase;

  static const TensorId Invalid;
};

inline constexpr TensorId TensorId::Invalid = TensorId(-1);

export struct DataId : IndexBase {
  using IndexBase::IndexBase;

  static const DataId Invalid;
};

inline constexpr DataId DataId::Invalid = DataId(-1);

}  // namespace axon
