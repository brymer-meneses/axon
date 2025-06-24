module;

#include <cassert>
#include <ranges>
#include <vector>

export module axon.storage;
import axon.ids;

namespace axon {

export template <typename IndexType, typename ValueType>
  requires(std::is_base_of_v<IndexBase<IndexType>, IndexType>)
class Storage {
 public:
  auto push_back(const ValueType& value) -> IndexType {
    values_.push_back(value);
    return IndexType(values_.size() - 1);
  }

  template <typename... Args>
  auto emplace_back(Args&&... args) -> IndexType {
    values_.emplace_back(std::forward<Args>(args)...);
    return IndexType(values_.size() - 1);
  }

  auto get(IndexType index) -> ValueType& {
    assert(index.has_value());
    return values_[index.value()];
  }

  auto get(IndexType index) const -> const ValueType& {
    assert(index.has_value());
    return values_[index.value()];
  }

  auto iter() {
    return std::views::iota(0u, values_.size()) |
           std::views::transform([](auto val) { return IndexType(val); });
  }

 private:
  std::vector<ValueType> values_;
};

}  // namespace axon
