module;

#include <cassert>
#include <ranges>
#include <vector>

#include "axon/base/dcheck.h"

export module axon.base:storage;

import :index_base;

export namespace axon {

template <Index IndexType, typename ValueType>
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
    AXON_DCHECK(index.has_value(), "Passed index has no value.");
    return values_[index.value()];
  }

  auto get(IndexType index) const -> const ValueType& {
    AXON_DCHECK(index.has_value(), "Passed index has no value.");
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
    AXON_DCHECK(index.has_value(), "Passed index has no value.");
    return static_cast<int32_t>(index.value()) < values_.size();
  }

 private:
  std::vector<ValueType> values_;
};

template <Index LeftIndexType, Index RightIndexType>
class RelationalStore {
 public:
  auto create_relation(LeftIndexType lhs, RightIndexType rhs) -> void {
    AXON_DCHECK(not contains_source(lhs),
                "Passed source has already an existing relation.");
    AXON_DCHECK(not contains_source(rhs),
                "Passed target has already an existing relation.");
    relations_.push_back({lhs, rhs});
  }

  auto contains_source(LeftIndexType lhs) const -> bool {
    for (const auto& value : relations_) {
      if (value.first == lhs) {
        return true;
      }
    }
    return false;
  }

  auto contains_target(RightIndexType rhs) const -> bool {
    for (const auto& value : relations_) {
      if (value.second == rhs) {
        return true;
      }
    }
    return false;
  }

  auto get_source(RightIndexType lhs) const -> LeftIndexType {
    for (const auto& value : relations_) {
      if (value.second == lhs) {
        return value.first;
      }
    }
    return LeftIndexType::None;
  }

  auto get_target(LeftIndexType lhs) const -> RightIndexType {
    for (const auto& value : relations_) {
      if (value.first == lhs) {
        return value.second;
      }
    }
    return RightIndexType::None;
  }

  auto size() const -> size_t { return relations_.size(); }

  auto relations() const -> const auto& { return relations_; }
  auto relations() -> auto& { return relations_; }

 private:
  std::vector<std::pair<LeftIndexType, RightIndexType>> relations_;
};

// TODO: use binary search here.
template <Index KeyType, Index ValueType>
class IdStore {
  struct KeyValuePair {
    KeyType key;
    ValueType value;
  };

 public:
  auto add(KeyType key, ValueType value) -> void {
    pairs_.emplace_back(key, value);
  }

  auto operator[](KeyType key) const -> const ValueType& {
    if (contains(key)) {
      return pairs_[key.value()].value;
    }

    add(key, ValueType::None);
    return pairs_[key.value()].value;
  }

  auto operator[](KeyType key) -> ValueType& {
    if (contains(key)) {
      return pairs_[key.value()].value;
    }

    add(key, ValueType::None);
    return pairs_[key.value()].value;
  }

  auto contains(KeyType key) const -> bool {
    AXON_DCHECK(key.has_value(), "Passed index has no value.");
    return static_cast<size_t>(key.value()) < pairs_.size();
  }

  auto get(KeyType target_key) -> ValueType {
    AXON_DCHECK(target_key.has_value(), "Passed key must have a value.");
    return pairs_[target_key.value()].value;
  }

  auto get(KeyType target_key) const -> ValueType {
    AXON_DCHECK(target_key.has_value(), "Passed key must have a value.");
    return pairs_[target_key.value()].value;
  }

  auto size() const -> size_t { return pairs_.size(); }

  auto keys() const -> auto {
    return std::views::transform(pairs_,
                                 [](KeyValuePair pair) { return pair.key; });
  }
  auto values() const -> auto {
    return std::views::transform(pairs_,
                                 [](KeyValuePair pair) { return pair.value; });
  }

 private:
  std::vector<KeyValuePair> pairs_;
};

}  // namespace axon
