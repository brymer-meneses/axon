module;

#include <cassert>
#include <ranges>
#include <vector>

export module axon.base:storage;

import :index_base;

export namespace axon {

template <typename IndexType, typename ValueType>
  requires(IsIndex<IndexType>)
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

template <typename LeftIndexType, typename RightIndexType>
  requires(IsIndex<LeftIndexType> and IsIndex<RightIndexType>)
class RelationalStore {
 public:
  auto create_relation(LeftIndexType lhs, RightIndexType rhs) -> void {
    assert(not contains_source(lhs) and not contains_target(rhs));
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
    return LeftIndexType::Invalid;
  }

  auto get_target(LeftIndexType lhs) const -> RightIndexType {
    for (const auto& value : relations_) {
      if (value.first == lhs) {
        return value.second;
      }
    }
    return RightIndexType::Invalid;
  }

  auto size() const -> size_t { return relations_.size(); }

  auto relations() const -> const auto& { return relations_; }
  auto relations() -> auto& { return relations_; }

 private:
  std::vector<std::pair<LeftIndexType, RightIndexType>> relations_;
};

}  // namespace axon
