module;

#include <memory>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/core/xshape.hpp"

export module axon.core:tensor;

import :ids;
import :storage;

export namespace axon {}  // namespace axon
