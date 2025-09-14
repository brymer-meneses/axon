module;

#include <ranges>
#include <vector>

#include "axon/base/macros.h"

export module axon.base:storage;

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

  auto operator==(const ValueStore& rhs) const -> bool = default;

 private:
  std::vector<ValueType> values_;
};

template <Index KeyType, Index ValueType>
class RelationalStore {
 public:
  auto operator==(const RelationalStore& rhs) const -> bool = default;

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

    auto operator==(const KeyValuePair& rhs) const -> bool = default;
  };

 public:
  auto operator==(const IdStore& rhs) const -> bool = default;

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

template <Index KeyType, typename ValueType>
class IdMap {
  using ValueTypeRef = std::reference_wrapper<ValueType>;
  using ValueTypeConstRef = std::reference_wrapper<const ValueType>;

 public:
  auto operator==(const IdMap& rhs) const -> bool = default;

  auto set(KeyType key, ValueType&& value) -> void {
    for (uint64_t i = 0; i < keys_.size(); i += 1) {
      if (keys_[i] == key) {
        values_[i] = std::move(value);
        return;
      }
    }

    keys_.emplace_back(key);
    values_.emplace_back(std::move(value));
  }

  auto set(KeyType key, const ValueType& value) -> void {
    for (uint64_t i = 0; i < keys_.size(); i += 1) {
      if (keys_[i] == key) {
        values_[i] = std::move(value);
        return;
      }
    }

    keys_.emplace_back(key);
    values_.emplace_back(std::move(value));
  }

  auto get(KeyType key) -> std::optional<ValueTypeRef> {
    for (uint64_t i = 0; i < keys_.size(); i += 1) {
      if (keys_[i] == key) {
        return std::ref(values_[i]);
      }
    }
    return std::nullopt;
  }

  auto get(KeyType key) const -> std::optional<ValueTypeConstRef> {
    for (uint64_t i = 0; i < keys_.size(); i += 1) {
      if (keys_[i] == key) {
        return std::cref(values_[i]);
      }
    }
    return std::nullopt;
  }

  auto size() const -> size_t { return keys_.size(); }

  auto keys() -> std::vector<KeyType>& { return keys_; }
  auto keys() const -> const std::vector<KeyType>& { return keys_; }

  auto values() -> std::vector<ValueType>& { return values_; }
  auto values() const -> const std::vector<ValueType>& { return values_; }

  auto pairs() -> auto { return std::views::zip(keys_, values_); }
  auto pairs() const -> auto { return std::views::zip(keys_, values_); }

 private:
  std::vector<KeyType> keys_;
  std::vector<ValueType> values_;
};

}  // namespace axon
