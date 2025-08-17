
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"

import axon.core;
import axon.python;

namespace nb = nanobind;

using namespace axon;

NB_MODULE(axon_bindings, m) {
  nb::class_<Tensor>(m, "Tensor")
      .def("__repr__",
           [](const Tensor& self) -> std::string { return "tensor"; });

  m.def(
      "create_tensor",
      [](nb::ndarray<>& array, bool requires_grad) -> Tensor {
        if (not requires_grad) {
          auto data = Storage::fromNanobind(array, ElementType::Float32);
          return {data};
        }

        auto data = Storage::fromNanobind(array, ElementType::Float32);
        auto grad = Storage::createZerosLike(data);
        return {data, grad};
      },
      nb::rv_policy::move);
}
