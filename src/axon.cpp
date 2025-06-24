module;

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xio.hpp"

export module axon;

import axon.graph;
import axon.tensor;
import axon.context;
import axon.inst;

namespace nb = nanobind;

static auto convert_to_xarray(nb::ndarray<nb::numpy, float> nb_array)
    -> xt::xarray<float> {
  std::vector<size_t> shape;
  for (auto i = 0; i < nb_array.ndim(); ++i) {
    shape.push_back(nb_array.shape(i));
  }

  auto xt_view = xt::adapt(static_cast<const float*>(nb_array.data()), shape);
  return xt::xarray<float>(xt_view);
}

NB_MODULE(_cpp, m) {
  static axon::Context context;

  nb::class_<axon::Tensor>(m, "Tensor")
      .def("__repr__",
           [](std::shared_ptr<axon::Tensor> tensor) {
             std::stringstream stream;

             context.executor()->forward();

             const auto& inst = stream << *tensor->data().storage;
             return stream.str();
           })
      .def(
          "__add__",
          [](std::shared_ptr<axon::Tensor> lhs,
             std::shared_ptr<axon::Tensor> rhs) {
            auto graph = context.graph();
            auto inst_id = graph->apply_operation<axon::insts::Add>(
                lhs->inst_id(), rhs->inst_id());
            return std::make_shared<axon::Tensor>(inst_id, graph);
          },
          nb::is_operator())
      .def(
          "__matmul__",
          [](std::shared_ptr<axon::Tensor> lhs,
             std::shared_ptr<axon::Tensor> rhs) {
            auto graph = context.graph();
            auto inst_id = graph->apply_operation<axon::insts::MatMul>(
                lhs->inst_id(), rhs->inst_id());
            return std::make_shared<axon::Tensor>(inst_id, graph);
          },
          nb::is_operator())
      .def(
          "__mul__",
          [](std::shared_ptr<axon::Tensor> lhs,
             std::shared_ptr<axon::Tensor> rhs) {
            auto graph = context.graph();
            auto inst_id = graph->apply_operation<axon::insts::Mul>(
                lhs->inst_id(), rhs->inst_id());
            return std::make_shared<axon::Tensor>(inst_id, graph);
          },
          nb::is_operator())
      .def_prop_ro("requires_grad", [](std::shared_ptr<axon::Tensor> tensor) {
        return tensor->requires_grad();
      });

  m.def("create_tensor",
        [](nb::ndarray<nb::numpy, float> raw_data,
           bool requires_grad) -> std::shared_ptr<axon::Tensor> {
          auto&& data = convert_to_xarray(raw_data);
          return std::make_shared<axon::Tensor>(
              context.create_tensor(data, requires_grad));
        });
}
