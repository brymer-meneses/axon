module;

#include <cassert>
#include <cstdint>

export module axon.core.ids;

import axon.base.index;

namespace axon {

export struct InstId : IndexBase<InstId> {
  using IndexBase::IndexBase;

  static const InstId Invalid;
  static const InstId Pending;
};

inline constexpr auto InstId::Invalid = InstId(-1);
inline constexpr auto InstId::Pending = InstId(-2);

export struct DataId : IndexBase<DataId> {
  using IndexBase::IndexBase;

  static const DataId Invalid;
};

inline constexpr auto DataId::Invalid = DataId(-1);

export struct ParamId : IndexBase<ParamId> {
  using IndexBase::IndexBase;

  static const ParamId Invalid;
};
inline constexpr auto ParamId::Invalid = ParamId(-1);

}  // namespace axon
