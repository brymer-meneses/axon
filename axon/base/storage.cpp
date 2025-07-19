module;

#include <cassert>
#include <ranges>
#include <utility>
#include <vector>

#include "axon/base/dcheck.h"
#include "llvm/ADT/STLExtras.h"

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

  auto keys() const -> auto {
    return std::views::transform(relations_,
                                 [](auto relation) { return relation.first; });
  }
  auto values() const -> auto {
    return std::views::transform(relations_,
                                 [](auto relation) { return relation.second; });
  }

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
    AXON_DCHECK(not contains(key), "Passed key must used.");
    pairs_.emplace_back(key, value);
  }

  auto contains(KeyType target_key) const -> bool {
    for (auto key : keys()) {
      if (target_key == key) {
        return true;
      }
    }
    return false;
  }

  auto operator[](KeyType key) const -> const ValueType& { return get(key); }
  auto operator[](KeyType key) -> ValueType& { return get(key); }

  auto get(KeyType target_key) -> ValueType& {
    AXON_DCHECK(target_key.has_value(), "Passed key must have a value.");
    for (auto& [key, value] : pairs_) {
      if (target_key == key) {
        return value;
      }
    }

    add(target_key, ValueType::None);
    return pairs_.back().value;
  }

  auto get(KeyType target_key) const -> const ValueType& {
    AXON_DCHECK(target_key.has_value(), "Passed key must have a value.");
    for (const auto& [key, value] : pairs_) {
      if (target_key == key) {
        return value;
      }
    }

    add(target_key, ValueType::None);
    return pairs_.back().value;
  }

  auto set(KeyType target_key, ValueType target_value) {
    AXON_DCHECK(target_key.has_value(), "Passed key must have a value.");
    for (auto& [i, pair] : llvm::enumerate(pairs_)) {
      if (pair[i].key == target_key) {
        pairs_[i].value = target_value;
      }
    }
    add(target_key, target_value);
  }

  auto size() const -> size_t { return pairs_.size(); }

  auto pairs() const -> const auto& { return pairs_; }
  auto pairs() -> auto& { return pairs_; }

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
