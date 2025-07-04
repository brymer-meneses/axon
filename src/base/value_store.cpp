module;

#include <cassert>
#include <ranges>
#include <vector>

export module axon.base.value_store;

import axon.base.index;

namespace axon {

export template <typename IndexType, typename ValueType>
  requires(std::is_base_of_v<IndexBase<IndexType>, IndexType>)
class ValueStore {
 public:
  auto add(const ValueType& value) -> IndexType {
    values_.push_back(value);
    return IndexType(values_.size() - 1);
  }

  template <typename... Args>
  auto emplace(Args&&... args) -> IndexType {
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

  auto size() const -> size_t { return values_.size(); }

  auto iter() const -> auto {
    return std::views::iota(0u, values_.size()) |
           std::views::transform([](auto val) { return IndexType(val); });
  }

  auto iter_values() const -> auto {
    return std::views::iota(0u, values_.size()) |
           std::views::transform(
               [this](auto val) -> std::pair<IndexType, const ValueType&> {
                 return {IndexType(val), values_[val]};
               });
  }

  auto iter_values() -> auto {
    return std::views::iota(0u, values_.size()) |
           std::views::transform(
               [this](auto val) -> std::pair<IndexType, ValueType&> {
                 return {IndexType(val), values_[val]};
               });
  }

  auto contains(IndexType index) const -> bool {
    assert(index.has_value());
    return static_cast<uint32_t>(index.value()) < values_.size();
  }

 private:
  std::vector<ValueType> values_;
};

}  // namespace axon
