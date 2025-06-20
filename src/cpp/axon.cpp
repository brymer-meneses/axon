module;

#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "xtensor/io/xio.hpp"

export module axon;

import axon.tensor;
import axon.global_context;

namespace nb = nanobind;

NB_MODULE(_cpp, m) {
  using namespace axon;

  nb::class_<Tensor>(m, "Tensor")
      .def("__repr__",
           [](std::shared_ptr<Tensor> self) {
             std::stringstream stream;
             auto& ctx = GlobalContext::Get();
             stream << "\n" << ctx.Data(self->id);

             return stream.str();
           })
      .def_prop_ro("requires_grad", [](std::shared_ptr<Tensor> self) -> bool {
        auto& ctx = GlobalContext::Get();
        return ctx.RequiresGrad(self->id);
      });

  m.def(
      "_create_tensor",
      [](const nb::ndarray<nb::numpy, float>& data, bool requires_grad) {
        auto& ctx = GlobalContext::Get();
        auto tensor_id = ctx.CreateTensor(data, requires_grad);
        return Tensor(tensor_id);
      },
      nb::rv_policy::move);
}
