module;

#include <cassert>

export module axon.core:ids;

import axon.base;

export namespace axon::core {

struct InstId : IndexBase<InstId> {
  using IndexBase::IndexBase;

  static const InstId Pending;

  auto is_pending() const -> bool { return *this == InstId::Pending; }
};

inline constexpr auto InstId::Pending = InstId(-2);

struct InputId : IndexBase<InputId> {
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

}  // namespace axon::core
