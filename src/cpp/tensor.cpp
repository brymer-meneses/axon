module;

#include <cassert>

export module axon.tensor;

import axon.ids;
import axon.global_context;

namespace axon {

// A `Tensor` is a lightweight "handle" to the `TensorMetadata` which lives
// inside the `GlobalContext`.
export class Tensor {
 public:
  Tensor(TensorId id) : id(id) {}

  ~Tensor() {
    auto& ctx = GlobalContext::Get();
    ctx.Destroy(id);
  }

  TensorId id;
};

}  // namespace axon
