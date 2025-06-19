module;

#include <cassert>
#include <memory>
#include <optional>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"

export module axon.tensor;

import axon.common;
import axon.context;

namespace nb = nanobind;

namespace axon {

// A `Tensor` is a lightweight "handle" to the `TensorMetadata` which lives
// inside a `Context`.
export class Tensor {
 public:
  Tensor(std::shared_ptr<Context> context, TensorId id)
      : context(context), id(id) {
    assert(id.is_valid());
  }

  std::shared_ptr<Context> context;
  TensorId id;
};

}  // namespace axon
