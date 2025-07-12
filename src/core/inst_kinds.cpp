module;

#include <cstdint>

#include "llvm/ADT/SmallVector.h"

export module axon.core:inst_kinds;

import :ids;

export namespace axon::insts {

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
  ArgumentId argument;
};

struct GetCachedValue {
  CachedValueId value_id;
};

struct SetCachedValue {
  InstId value;
  CachedValueId value_id;
};

struct Constant {};

}  // namespace axon::insts
