module;

#include <cassert>
#include <cstdint>

export module axon.core:ids;

import axon.base;

export namespace axon {

struct InstId : IndexBase<InstId> {
  using IndexBase::IndexBase;

  static const InstId Invalid;
  static const InstId Pending;
};
inline constexpr auto InstId::Invalid = InstId(-1);
inline constexpr auto InstId::Pending = InstId(-2);

struct ArgumentId : IndexBase<ArgumentId> {
  using IndexBase::IndexBase;

  static const ArgumentId Invalid;
};
inline constexpr auto ArgumentId::Invalid = ArgumentId(-1);

struct TensorId : IndexBase<TensorId> {
  using IndexBase::IndexBase;

  static const TensorId Invalid;
};
inline constexpr auto TensorId::Invalid = TensorId(-1);

struct ForeignTensorId : IndexBase<ForeignTensorId> {
  using IndexBase::IndexBase;

  static const ForeignTensorId Invalid;
};
inline constexpr auto ForeignTensorId::Invalid = ForeignTensorId(-1);

struct CachedValueId : IndexBase<CachedValueId> {
  using IndexBase::IndexBase;

  static const CachedValueId Invalid;
};

inline constexpr auto CachedValueId::Invalid = CachedValueId(-1);

}  // namespace axon
