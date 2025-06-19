module;

#include <cassert>
#include <cstdint>
#include <vector>

export module axon.common;

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

export template <typename IndexType, typename ValueType>
  requires(std::is_base_of_v<IndexBase, IndexType>)
class BasicStorage {
 public:
  constexpr auto operator[](IndexType index) const -> const ValueType& {
    assert(index.is_valid());
    return storage_[*index];
  }

  constexpr auto operator[](IndexType index) -> ValueType& {
    assert(index.is_valid());
    return storage_[*index];
  }

  constexpr auto Append(ValueType&& value) -> IndexType {
    storage_.push_back(value);
    return IndexType(storage_.size() - 1);
  };

 private:
  std::vector<ValueType> storage_;
};

};  // namespace axon
