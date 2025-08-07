module;

#include "axon/base/dcheck.h"
#include "llvm/ADT/STLExtras.h"

export module axon.base:storage;

import std;

import :index_base;

export namespace axon {

template <Index KeyType, typename ValueType>
class ValueStore {
 public:
  auto add(const ValueType& value) -> KeyType {
    values_.push_back(value);
    return KeyType(values_.size() - 1);
  }

  template <typename... Args>
  auto emplace(Args&&... args) -> KeyType {
    values_.emplace_back(std::forward<Args>(args)...);
    return KeyType(values_.size() - 1);
  }

  auto get(KeyType index) -> ValueType& {
    AXON_DCHECK(index.isValid(), "Passed index has no value.");
    return values_[index.value()];
  }

  auto get(KeyType index) const -> const ValueType& {
    AXON_DCHECK(index.isValid(), "Passed index has no value.");
    return values_[index.value()];
  }

  auto size() const -> size_t { return values_.size(); }

  auto keys() const -> auto {
    return std::views::iota(0u, values_.size()) |
           std::views::transform([](auto val) { return KeyType(val); });
  }

  auto keysAndValues() const -> auto {
    return std::views::iota(0u, values_.size()) |
           std::views::transform(
               [this](auto val) -> std::pair<KeyType, const ValueType&> {
                 return {KeyType(val), values_[val]};
               });
  }

  auto keysAndValues() -> auto {
    return std::views::iota(0u, values_.size()) |
           std::views::transform(
               [this](auto val) -> std::pair<KeyType, ValueType&> {
                 return {KeyType(val), values_[val]};
               });
  }

  auto values() -> auto& { return values_; }
  auto values() const -> const auto& { return values_; }

  auto containsKey(KeyType index) const -> bool {
    AXON_DCHECK(index.isValid(), "Passed index has no value.");
    return static_cast<int32_t>(index.value()) < values_.size();
  }

 private:
  std::vector<ValueType> values_;
};

template <Index KeyType, Index ValueType>
class RelationalStore {
 public:
  auto createRelation(KeyType lhs, ValueType rhs) -> void {
    AXON_DCHECK(not containsKey(lhs),
                "Passed source has already an existing relation.");
    AXON_DCHECK(not containsKey(rhs),
                "Passed target has already an existing relation.");
    relations_.push_back({lhs, rhs});
  }

  auto containsKey(KeyType lhs) const -> bool {
    for (const auto& value : relations_) {
      if (value.first == lhs) {
        return true;
      }
    }
    return false;
  }

  auto containsValue(ValueType rhs) const -> bool {
    for (const auto& value : relations_) {
      if (value.second == rhs) {
        return true;
      }
    }
    return false;
  }

  auto getKeyOf(ValueType lhs) const -> KeyType {
    for (const auto& value : relations_) {
      if (value.second == lhs) {
        return value.first;
      }
    }
    return KeyType::None;
  }

  auto getValueOf(KeyType lhs) const -> ValueType {
    for (const auto& value : relations_) {
      if (value.first == lhs) {
        return value.second;
      }
    }
    return ValueType::None;
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
  std::vector<std::pair<KeyType, ValueType>> relations_;
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
    AXON_DCHECK(not containsKey(key), "Passed key must not be used.");
    pairs_.emplace_back(key, value);
  }

  auto containsKey(KeyType target_key) const -> bool {
    for (auto pair : pairs_) {
      if (target_key == pair.key) {
        return true;
      }
    }
    return false;
  }

  auto get(KeyType target_key) const -> ValueType {
    AXON_DCHECK(target_key.isValid(), "Passed key must have a value.");
    for (auto [key, value] : pairs_) {
      if (target_key == key) {
        return value;
      }
    }

    return ValueType::None;
  }

  auto set(KeyType target_key, ValueType target_value) -> void {
    AXON_DCHECK(target_key.isValid(), "Passed key must have a value.");
    for (auto& pair : pairs_) {
      if (pair.key == target_key) {
        pair.value = target_value;
        return;
      }
    }

    add(target_key, target_value);
  }

  auto size() const -> size_t { return pairs_.size(); }

  auto pairs() const -> const auto& { return pairs_; }

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
