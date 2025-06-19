module;

#include <vector>

#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/unique_ptr.h"
#include "xtensor/io/xio.hpp"

export module axon;

import axon.tensor;
import axon.context;

namespace nb = nanobind;

NB_MODULE(_cpp, m) {
  nb::class_<axon::Tensor>(m, "Tensor")
      .def("__repr__", [](std::shared_ptr<axon::Tensor> tensor) {
        std::stringstream stream;
        stream << "\n" << tensor->context->GetData(tensor->id);

        return stream.str();
      });

  nb::class_<axon::Context>(m, "Context")
      .def(nb::init<>())
      .def(
          "__enter__",
          [](std::shared_ptr<axon::Context> self) {
            axon::Context::SetCurrent(self);
            return self;
          },
          nb::rv_policy::automatic_reference)
      .def("__exit__",
           [](std::shared_ptr<axon::Context> self, nb::args args) {
             axon::Context::SetCurrent(nullptr);
             return nb::none();
           })
      .def(
          "_create_tensor",
          [](std::shared_ptr<axon::Context> self,
             const nb::ndarray<nb::numpy, float>& data,
             bool requires_grad) -> std::shared_ptr<axon::Tensor> {
            auto tensor_id = self->CreateTensor(data, requires_grad);
            return std::make_shared<axon::Tensor>(self, tensor_id);
          },
          nb::rv_policy::automatic_reference)
      .def_static("_get_current", &axon::Context::GetCurrent,
                  nb::rv_policy::automatic_reference);
}
