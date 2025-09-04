#include <format>
#include <iostream>
#include <memory>
#include <print>
#include <ranges>
#include <sstream>
#include <stdexcept>

#include "axon/base/dcheck.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/OperationSupport.h"
#include "nanobind/intrusive/counter.h"
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

static auto createStoragefromNanobind(nb::ndarray<>& array,
                                      ElementType element_type) -> Storage {
  // TODO: Explore ways to avoid copying the memory.
  auto buffer_size = array.size() * element_type.getSizeInBytes();
  auto* data_ptr = new std::byte[buffer_size];
  std::memcpy(data_ptr, array.data(), buffer_size);

  auto shape = llvm::ArrayRef<int64_t>(array.shape_ptr(), array.ndim());
  auto stride = llvm::ArrayRef<int64_t>(array.stride_ptr(), array.ndim());

  AXON_DCHECK(element_type == ElementType::Float32,
              "Only float32 is supported for now.");

  return {element_type, data_ptr, shape, stride, /*is_owned=*/true};
}

struct BroadcastInfo {
  llvm::SmallVector<insts::Broadcast::Expansion> expansions;
  llvm::SmallVector<int64_t> unsqueezed_shape;
};

static auto tryGetBroadcastInfo(llvm::ArrayRef<int64_t> source_shape,
                                llvm::ArrayRef<int64_t> target_shape)
    -> std::optional<BroadcastInfo> {
  auto source_rank = static_cast<int64_t>(source_shape.size());
  auto target_rank = static_cast<int64_t>(target_shape.size());

  AXON_DCHECK(source_rank < target_rank, "");

  BroadcastInfo broadcast_info;

  for (auto i = target_rank - 1; i >= 0; i -= 1) {
    auto source_dim =
        i > source_rank - 1 ? 1 : source_shape[source_rank - 1 - i];
    auto target_dim = target_shape[target_rank - 1 - i];

    broadcast_info.unsqueezed_shape.push_back(source_dim);

    if (source_dim == 1) {
      broadcast_info.expansions.push_back(
          {.dim = target_rank - 1 - i, .scale = target_dim});
      continue;
    }

    // if `source_dim` and `target_dim` are not equal and `source_dim` is not
    // equal to 1 then we cannot broadcast `source_shape` into the
    // `target_shape`.
    if (source_dim != target_dim) {
      return std::nullopt;
    }
  }

  return broadcast_info;
}

static auto performBroadcasting(Graph& graph, InstId source_id,
                                llvm::ArrayRef<int64_t> source_shape,
                                llvm::ArrayRef<int64_t> target_shape)
    -> InstId {
  auto broadcast_info = tryGetBroadcastInfo(source_shape, target_shape);
  if (!broadcast_info) {
    throw std::runtime_error(std::format("Failed to broadcast {} into {}",
                                         source_shape, target_shape));
  }

  auto reshaped_id = graph.createOp(
      insts::Reshape(source_id, broadcast_info->unsqueezed_shape));
  return graph.createOp(
      insts::Broadcast(reshaped_id, broadcast_info->expansions));
}

static thread_local std::weak_ptr<Graph> current_graph;

static auto buildGraphBindings(nb::module_& m) -> void {
  struct GraphRef : nb::intrusive_base {
    std::shared_ptr<Graph> inner{};

    GraphRef() : inner(std::make_shared<Graph>()) {}
  };

  nb::class_<GraphRef>(m, "Graph")
      .def(nb::init<>())
      .def("compile",
           [](GraphRef& graph,
              LoweringLevel level) -> std::unique_ptr<CompilationUnit> {
             auto compilation_unit = std::make_unique<CompilationUnit>();
             if (compilation_unit->compile(*graph.inner, level).failed()) {
               throw std::runtime_error("Failed to compile graph");
             }
             return std::move(compilation_unit);
           })
      .def("create_constant",
           [](GraphRef& graph, nb::ndarray<>& array,
              ElementType::InternalType element_type) -> Tensor {
             auto data = createStoragefromNanobind(array, element_type);
             auto inst_id = graph.inner->createConstant(std::move(data));
             return {inst_id};
           })
      .def("trace", [](GraphRef& graph, Tensor& tensor) {
        AXON_DCHECK(tensor.hasData(), "Passed tensor must have data.");
        tensor.trace(*graph.inner);
      });

  m.def("_set_current_graph",
        [](GraphRef graph) -> void { current_graph = graph.inner; });
}

static auto buildTensorBindings(nb::module_& m) -> void {
  nb::class_<Tensor>(m, "Tensor")
      .def_prop_ro("shape",
                   [](const Tensor& self) -> std::vector<int64_t> {
                     return self.shape();
                   })
      .def_prop_ro(
          "requires_grad",
          [](const Tensor& self) -> bool { return self.requiresGrad(); })
      .def("__repr__",
           [](const Tensor& self) -> std::string {
             if (not self.hasData()) {
               return "LazyTensor";
             }
             return dumpTensor(self);
           })
      .def("__add__",
           [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
             if (auto graph = current_graph.lock()) {
               auto lhs_id = lhs.inst_id();
               auto rhs_id = rhs.inst_id();

               auto lhs_shape = graph->getShape(lhs_id);
               auto rhs_shape = graph->getShape(rhs_id);

               // try broadcast lhs to have the same shape as rhs
               if (lhs_shape.size() < rhs_shape.size()) {
                 lhs_id =
                     performBroadcasting(*graph, lhs_id, lhs_shape, rhs_shape);
               }

               // try broadcast lhs to have the same shape as rhs
               if (lhs_shape.size() > rhs_shape.size()) {
                 rhs_id =
                     performBroadcasting(*graph, rhs_id, rhs_shape, lhs_shape);
               }

               auto inst_id = graph->createOp(insts::Add(lhs_id, rhs_id));
               return {inst_id};
             }
             throw std::runtime_error(
                 "Failed to invoke add since there is no active "
                 "graph.");
           })
      .def("__mul__",
           [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
             if (auto graph = current_graph.lock()) {
               auto lhs_id = lhs.inst_id();
               auto rhs_id = rhs.inst_id();

               auto lhs_shape = graph->getShape(lhs_id);
               auto rhs_shape = graph->getShape(rhs_id);

               // try broadcast lhs to have the same shape as rhs
               if (lhs_shape.size() < rhs_shape.size()) {
                 lhs_id =
                     performBroadcasting(*graph, lhs_id, lhs_shape, rhs_shape);
               }

               // try broadcast lhs to have the same shape as rhs
               if (lhs_shape.size() > rhs_shape.size()) {
                 rhs_id =
                     performBroadcasting(*graph, rhs_id, rhs_shape, lhs_shape);
               }

               auto inst_id = graph->createOp(insts::Mul(lhs_id, rhs_id));
               return {inst_id};
             }
             throw std::runtime_error(
                 "Failed to invoke mul since there is no active "
                 "graph.");
           })
      .def("__matmul__",
           [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
             if (auto graph = current_graph.lock()) {
               auto lhs_id = lhs.inst_id();
               auto rhs_id = rhs.inst_id();

               auto inst_id = graph->createOp(insts::MatMul(lhs_id, rhs_id));

               return {inst_id};
             }

             throw std::runtime_error(
                 "Failed to invoke matmul since there is no active "
                 "graph.");
           })
      .def("backward",
           [](const Tensor& self) -> void {
             if (auto graph = current_graph.lock()) {
               axon::backward(*graph, self.inst_id());
               return;
             }

             throw std::runtime_error(
                 "Failed to invoke backpropagation since there is no active "
                 "graph.");
           })
      .def_prop_ro("grad",
                   [](const Tensor& self) -> std::shared_ptr<Tensor> {
                     return self.grad();
                   })
      .def_prop_ro("shape", [](const Tensor& self) -> std::vector<int64_t> {
        if (auto graph = current_graph.lock()) {
          return graph->getShape(self.inst_id());
        }

        return self.shape().vec();
      });

  m.def(
      "_create_tensor",
      [](nb::ndarray<>& array, bool requires_grad,
         ElementType::InternalType element_type) -> Tensor {
        auto tensor = Tensor(createStoragefromNanobind(array, element_type),
                             requires_grad);
        return tensor;
      },
      nb::rv_policy::move);
}

NB_MODULE(axon_bindings, m) {
  nb::enum_<ElementType::InternalType>(m, "ElementType")
      .value("Float32", ElementType::Float32)
      .value("Float64", ElementType::Float64)
      .export_values();

  nb::enum_<LoweringLevel>(m, "LoweringLevel")
      .value("Axon", LoweringLevel::Axon)
      .value("Standard", LoweringLevel::Standard)
      .value("Bufferization", LoweringLevel::Bufferization)
      .value("Affine", LoweringLevel::Affine)
      .value("Linalg", LoweringLevel::Linalg)
      .value("LLVM", LoweringLevel::LLVM)
      .export_values();

  nb::class_<CompilationUnit>(m, "CompilationUnit")
      .def("dump_ir",
           [](const CompilationUnit& self) { return self.dump_ir(); })
      .def("execute", [](CompilationUnit& self,
                         std::vector<std::shared_ptr<Tensor>> tensors) {
        self.execute(std::move(tensors));
      });

  buildTensorBindings(m);
  buildGraphBindings(m);
}
