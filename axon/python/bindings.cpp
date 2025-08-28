

#include <iostream>
#include <memory>
#include <ranges>
#include <sstream>
#include <stdexcept>

#include "axon/base/dcheck.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/OperationSupport.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/unique_ptr.h"
#include "nanobind/stl/vector.h"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xio.hpp"

import axon.core;
import axon.python;
import axon.mlir;

namespace nb = nanobind;

using namespace axon;

static thread_local std::shared_ptr<Graph> current_graph{};

static auto createStoragefromNanobind(nb::ndarray<>& array,
                                      ElementType element_type) -> Storage {
  // TODO: Explore ways to avoid copying the memory.
  auto* data_ptr = new std::byte[array.size()];
  std::memcpy(data_ptr, array.data(),
              element_type.getSizeInBytes() * array.size());

  auto shape = llvm::ArrayRef<int64_t>(array.shape_ptr(), array.ndim());
  auto stride = llvm::ArrayRef<int64_t>(array.stride_ptr(), array.ndim());

  AXON_DCHECK(element_type == ElementType::Float32,
              "Only float32 is supported for now.");

  return {element_type, data_ptr, shape, stride, /*is_owned=*/true};
}

NB_MODULE(axon_bindings, m) {
  auto lowering_ops = nb::class_<LoweringOps>(m, "LoweringOps");
  lowering_ops.def(nb::init<LoweringOps::Level>());
  lowering_ops.def_rw("level", &LoweringOps::level);

  nb::enum_<LoweringOps::Level>(lowering_ops, "Level")
      .value("Axon", LoweringOps::Level::Axon)
      .value("Standard", LoweringOps::Level::Standard)
      .value("LLVM", LoweringOps::Level::LLVM)
      .export_values();

  nb::enum_<ElementType::InternalType>(m, "ElementType")
      .value("Float32", ElementType::Float32)
      .value("Float64", ElementType::Float64)
      .export_values();

  nb::class_<Tensor>(m, "Tensor")
      .def_prop_ro("shape",
                   [](const Tensor& self) -> std::vector<int64_t> {
                     return self.shape();
                   })
      .def_prop_ro(
          "requires_grad",
          [](const Tensor& self) -> bool { return self.requiresGrad(); })
      .def("__repr__", [](const Tensor& self) -> std::string {
        std::stringstream stream{};

        stream << "[";

        for (auto i : std::views::iota(0, 3)) {
          float elem = reinterpret_cast<float*>(self.data.data())[i];
          stream << elem << ",";
        }

        stream << "]";

        stream << " [";
        if (self.requiresGrad()) {
          for (auto i : std::views::iota(0, 3)) {
            float elem = reinterpret_cast<float*>(self.grad->data())[i];
            stream << elem << ",";
          }
        }
        stream << "]";

        return stream.str();
      });

  m.def(
      "_create_tensor",
      [](nb::ndarray<>& array, bool requires_grad,
         ElementType::InternalType element_type) -> Tensor {
        if (not requires_grad) {
          auto data = createStoragefromNanobind(array, element_type);
          return {data};
        }

        auto data = createStoragefromNanobind(array, element_type);
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
      .def("create_constant",
           [](nb::ndarray<>& array,
              ElementType::InternalType element_type) -> LazyTensor {
             auto data = createStoragefromNanobind(array, element_type);
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
