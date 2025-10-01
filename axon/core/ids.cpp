module;

#include <cassert>

export module axon.core:ids;

import axon.base;

export namespace axon {

struct InstId : IndexBase<InstId> {
  using IndexBase::IndexBase;

  static const InstId Pending;

  constexpr auto add_offset(i32 offset) const -> InstId {
    if (isValid()) {
      return InstId(value() + offset);
    }
    return *this;
  }
};

inline constexpr auto InstId::Pending = InstId(-2);

struct ParamId : IndexBase<ParamId> {
  using IndexBase::IndexBase;

  constexpr auto add_offset(i32 offset) const -> ParamId {
    if (isValid()) {
      return ParamId(value() + offset);
    }
    return *this;
  }
};

}  // namespace axon
