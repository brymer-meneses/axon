module;

#include <cassert>

export module axon.core:ids;

import axon.base;

export namespace axon {

struct InstId : IndexBase<InstId> {
  using IndexBase::IndexBase;

  static const InstId Pending;
};

inline constexpr auto InstId::Pending = InstId(-2);

struct InputId : IndexBase<InputId> {
  using IndexBase::IndexBase;
};

struct BufferId : IndexBase<BufferId> {
  using IndexBase::IndexBase;
};

struct DataId : IndexBase<DataId> {
  using IndexBase::IndexBase;

  static const DataId Pending;
};
inline constexpr auto DataId::Pending = DataId(-2);

struct CachedValueId : IndexBase<CachedValueId> {
  using IndexBase::IndexBase;
};

}  // namespace axon
