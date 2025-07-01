module;

#include <optional>
#include <type_traits>
#include <vector>

export module axon.base.index_map;

import axon.base.index;

namespace axon {

export template <typename IndexType, typename ValueType>
  requires std::is_base_of_v<IndexBase<IndexType>, IndexType>
class IndexMap {
 public:
  auto get(IndexType index) const -> const ValueType* {
    for (const auto& val : values_) {
      if (val.first == index) {
        return &val.second;
      }
    }
    return nullptr;
  }

  auto get(IndexType index) -> ValueType* {
    for (auto& val : values_) {
      if (val.first == index) {
        return &val.second;
      }
    }
    return nullptr;
  }

  auto add(IndexType index, const ValueType& value) -> void {
    values_.emplace_back(index, value);
  }

 private:
  std::vector<std::pair<IndexType, ValueType>> values_;
};
}  // namespace axon
