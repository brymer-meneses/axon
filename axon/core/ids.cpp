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

struct ParamId : IndexBase<ParamId> {
  using IndexBase::IndexBase;
};

}  // namespace axon
