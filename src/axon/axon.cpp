module;

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

export module axon;
export import axon.op;
export import axon.tensor;
export import axon.context;

namespace nb = nanobind;

NB_MODULE(_axon_cpp, m) {
  nb::class_<axon::Context>(m, "Context")
      .def(nb::init<std::string>())
      .def("record_batch_matmul", &axon::Context::record_batch_matmul)
      .def("declare_tensor", &axon::Context::declare_tensor);

  nb::class_<axon::TensorId>(m, "TensorId").def(nb::init<int32_t>());
}
