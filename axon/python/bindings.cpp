

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/variant.h"

import axon.core;
import axon.python;

namespace nb = nanobind;

using namespace axon;

enum class InstType { Add, Mul };

NB_MODULE(axon_bindings, m) {
  nb::class_<Tensor>(m, "Tensor")
      .def("__repr__",
           [](const Tensor& self) -> std::string { return "tensor"; });

  nb::class_<LazyTensor>(m, "LazyTensor");

  nb::class_<Graph>(m, "Graph")
      .def(
          "create_op",
          [](Graph& self, InstType inst_type,
             std::vector<LazyTensor> params) -> LazyTensor {
            switch (inst_type) {
              case InstType::Add:
                return {self.createOp(
                    insts::Add(params[0].inst_id, params[1].inst_id))};
              case InstType::Mul:
                return {self.createOp(
                    insts::Mul(params[0].inst_id, params[1].inst_id))};
            };
          },
          nb::rv_policy::move)
      .def(
          "declare_parameter",
          [](Graph& self, std::vector<int64_t> shape,
             bool requires_grad) -> LazyTensor {
            return {
                self.declareParam({shape.begin(), shape.end()}, requires_grad)};
          },
          nb::rv_policy::move);

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
