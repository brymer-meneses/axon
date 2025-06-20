module;

#include <cassert>
#include <vector>

export module axon.storage;

import axon.ids;

namespace axon {

export template <typename IndexType, typename ValueType>
  requires(std::is_base_of_v<IndexBase, IndexType>)
class Storage {
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
