module;

#include <cassert>
#include <cstdint>

#include "llvm/ADT/SmallVector.h"

export module axon.common;

namespace axon {

export template <typename T, typename V>
class IdBase {
 public:
  using ValueType = V;

  constexpr auto is_valid() const -> bool { return value_ != -1; }
  constexpr auto operator*() const -> int32_t {
    assert(value_ != -1);
    return value_;
  }
  constexpr explicit IdBase(int32_t value) : value_(value) {}

 private:
  int32_t value_;
};

export template <typename IndexType>
class BasicStorage {
 public:
  using ValueType = typename IndexType::ValueType;

  constexpr auto operator[](IndexType index) const -> const ValueType& {
    assert(index.is_valid());
    return storage_[*index];
  }

  constexpr auto operator[](IndexType index) -> ValueType& {
    assert(index.is_valid());
    return storage_[*index];
  }

  constexpr auto append(ValueType value) -> IndexType {
    storage_.push_back(std::move(value));
    return IndexType(storage_.size() - 1);
  };

 private:
  llvm::SmallVector<ValueType> storage_;
};

};  // namespace axon
