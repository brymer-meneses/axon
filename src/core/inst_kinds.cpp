module;

#include <cstdint>

#include "llvm/ADT/SmallVector.h"

export module axon.core:inst_kinds;

import :ids;

namespace axon::insts {

struct MatMul {
  InstId lhs_id;
  InstId rhs_id;
};

struct Transpose {
  InstId value_id;
};

struct Add {
  InstId lhs_id;
  InstId rhs_id;
};

struct Mul {
  InstId lhs_id;
  InstId rhs_id;
};

struct GetFunctionArgument {
  int32_t argument;
};

struct GetCachedValue {
  int32_t value_id;
};

struct Write {
  InstId inst_id;
  CachedValueId value_id;
};

struct Constant {};

}  // namespace axon::insts
