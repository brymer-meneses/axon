module;

#include <cassert>
#include <cstdint>

export module axon.core.ids;

import axon.base.index;

export namespace axon {

struct InstId : IndexBase<InstId> {
  using IndexBase::IndexBase;

  static const InstId Invalid;
  static const InstId Pending;
};

inline constexpr auto InstId::Invalid = InstId(-1);
inline constexpr auto InstId::Pending = InstId(-2);

struct ParamId : IndexBase<ParamId> {
  using IndexBase::IndexBase;

  static const ParamId Invalid;
};
inline constexpr auto ParamId::Invalid = ParamId(-1);

struct CachedValueId : IndexBase<CachedValueId> {
  using IndexBase::IndexBase;

  static const CachedValueId Invalid;
};

inline constexpr auto CachedValueId::Invalid = CachedValueId(-1);

}  // namespace axon
