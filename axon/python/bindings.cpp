module;

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"

export module axon.python;

import axon.core;

import :tensor;
import :storage;

namespace nb = nanobind;

using namespace axon;

NB_MODULE(_cpp, m) {
  nb::class_<Tensor>(m, "Tensor")
      .def("__init__",
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
