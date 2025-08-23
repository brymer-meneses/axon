

#include <memory>
#include <sstream>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/OperationSupport.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/unique_ptr.h"
#include "nanobind/stl/vector.h"

import axon.core;
import axon.python;

namespace nb = nanobind;

using namespace axon;

static thread_local std::shared_ptr<Graph> current_graph{};

NB_MODULE(axon_bindings, m) {
  nb::class_<Tensor>(m, "Tensor")
      .def("__repr__",
           [](const Tensor& self) -> std::string { return "tensor"; })
      .def_prop_ro("shape",
                   [](const Tensor& self) -> std::vector<int64_t> {
                     return self.shape();
                   })
      .def_prop_ro("requires_grad", [](const Tensor& self) -> bool {
        return self.requiresGrad();
      });

  nb::class_<LazyTensor>(m, "LazyTensor")
      .def("__add__",
           [](const LazyTensor& lhs, const LazyTensor& rhs) -> LazyTensor {
             auto inst_id =
                 current_graph->createOp(insts::Add(lhs.inst_id, rhs.inst_id));
             return {inst_id};
           })
      .def("__mul__",
           [](const LazyTensor& lhs, const LazyTensor& rhs) -> LazyTensor {
             auto inst_id =
                 current_graph->createOp(insts::Mul(lhs.inst_id, rhs.inst_id));
             return {inst_id};
           });
  nb::class_<CompilationUnit>(m, "CompilationUnit")
      .def("compile",
           [](CompilationUnit& self, Graph& graph) { self.compile(graph); })
      .def("__repr__", [](CompilationUnit& self) -> std::string {
        mlir::OpPrintingFlags flags;
        std::string repr;
        llvm::raw_string_ostream string_stream{repr};
        self.module_op()->print(string_stream, flags);
        return repr;
      });

  nb::class_<Graph>(m, "Graph")
      .def(nb::init<>())
      .def(
          "declare_parameter",
          [](Graph& self, std::vector<int64_t> shape,
             bool requires_grad) -> LazyTensor {
            llvm::SmallVector<int64_t> shape_ = {shape.begin(), shape.end()};
            auto inst_id = self.declareParam(shape_, requires_grad);
            return {inst_id};
          },
          nb::rv_policy::move)
      .def("finalize",
           [](Graph& self, LazyTensor tensor) {
             axon::backward(self, tensor.inst_id);
           })
      .def("compile", [](Graph& graph) -> std::unique_ptr<CompilationUnit> {
        auto compilation_unit = std::make_unique<CompilationUnit>();
        compilation_unit->compile(graph);
        return std::move(compilation_unit);
      });

  m.def(
      "_create_tensor",
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

  m.def("_get_current_graph",
        []() -> std::shared_ptr<Graph> { return current_graph; });

  m.def("_set_current_graph", [](std::shared_ptr<Graph> graph) -> void {
    current_graph = std::move(graph);
  });
}
