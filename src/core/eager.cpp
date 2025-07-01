module;

#include <memory>
#include <optional>
#include <ranges>
#include <variant>

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/containers/xarray.hpp"

export module axon.core.eager;

import axon.core.graph;
import axon.core.inst;
import axon.core.ids;

namespace axon {}  // namespace axon
