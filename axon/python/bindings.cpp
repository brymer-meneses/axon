

#include <memory>
#include <ranges>
#include <sstream>
#include <stdexcept>

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
import axon.mlir;

namespace nb = nanobind;

using namespace axon;

static thread_local std::shared_ptr<Graph> current_graph{};

static auto createStoragefromNanobind(nb::ndarray<>& array,
                                      ElementType element_type) -> Storage {
  auto* data = reinterpret_cast<std::byte*>(array.data());
  auto shape = llvm::ArrayRef<int64_t>(array.shape_ptr(), array.ndim());
  auto stride = llvm::ArrayRef<int64_t>(array.stride_ptr(), array.ndim());

  return {element_type, data, shape, stride, /*is_owned=*/false};
}

NB_MODULE(axon_bindings, m) {
  nb::class_<Tensor>(m, "Tensor")
      .def_prop_ro("shape",
                   [](const Tensor& self) -> std::vector<int64_t> {
                     return self.shape();
                   })
      .def_prop_ro("requires_grad", [](const Tensor& self) -> bool {
        return self.requiresGrad();
      });

  m.def(
      "_create_tensor",
      [](nb::ndarray<>& array, bool requires_grad) -> Tensor {
        if (not requires_grad) {
          auto data = createStoragefromNanobind(array, ElementType::Float32);
          return {data};
        }

        auto data = createStoragefromNanobind(array, ElementType::Float32);
        auto grad = Storage::createZerosLike(data);
        return {data, grad};
      },
      nb::rv_policy::move);

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
           })
      .def("backward", [](const LazyTensor& self) -> void {
        axon::backward(*current_graph, self.inst_id);
      });

  nb::class_<CompilationUnit>(m, "CompilationUnit")
      .def("dump_ir",
           [](const CompilationUnit& self) { return self.dump_ir(); })
      .def("execute", [](CompilationUnit& self,
                         std::vector<std::shared_ptr<Tensor>> tensors) {
        self.execute(std::move(tensors));
      });

  auto lowering_ops = nb::class_<LoweringOps>(m, "LoweringOps");
  lowering_ops.def(nb::init<LoweringOps::Level>());
  lowering_ops.def_rw("level", &LoweringOps::level);

  nb::enum_<LoweringOps::Level>(lowering_ops, "Level")
      .value("Axon", LoweringOps::Level::Axon)
      .value("Standard", LoweringOps::Level::Standard)
      .value("LLVM", LoweringOps::Level::LLVM)
      .export_values();

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
      .def("compile",
           [](Graph& graph,
              const LoweringOps& ops) -> std::unique_ptr<CompilationUnit> {
             auto compilation_unit = std::make_unique<CompilationUnit>();
             if (compilation_unit->compile(graph, ops).failed()) {
               throw std::runtime_error("Failed to compile graph");
             }
             return std::move(compilation_unit);
           })
      .def("create_constant", [](nb::ndarray<>& array) -> LazyTensor {
        auto data = createStoragefromNanobind(array, ElementType::Float32);
        auto inst_id = current_graph->createConstant(std::move(data));
        return {inst_id};
      });

  m.def("_get_current_graph",
        []() -> std::shared_ptr<Graph> { return current_graph; });

  m.def("_set_current_graph", [](std::shared_ptr<Graph> graph) -> void {
    current_graph = std::move(graph);
  });

  m.def("_clear_current_graph", []() -> void { current_graph.reset(); });
}
