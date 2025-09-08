#include <format>
#include <memory>
#include <ranges>
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

static auto createStoragefromNanobind(nb::ndarray<>& array, DataType data_type)
    -> Storage {
  // TODO: Explore ways to avoid copying the memory.
  auto buffer_size = array.size() * data_type.getSizeInBytes();
  auto* data_ptr = new std::byte[buffer_size];
  std::memcpy(data_ptr, array.data(), buffer_size);

  auto shape = llvm::ArrayRef<int64_t>(array.shape_ptr(), array.ndim());
  auto stride = llvm::ArrayRef<int64_t>(array.stride_ptr(), array.ndim());

  AXON_DCHECK(data_type == DataType::Float32,
              "Only float32 is supported for now.");

  return Storage::create(data_ptr, shape, data_type, /*is_owned=*/true, stride);
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
              DataType::InternalType data_type) -> Tensor {
             auto data = createStoragefromNanobind(array, data_type);
             auto inst_id = graph.inner->createConstant(std::move(data));
             return {inst_id};
           })
      .def("trace",
           [](GraphRef& graph, Tensor& tensor) {
             AXON_DCHECK(tensor.hasData(), "Passed tensor must have data.");
             tensor.trace(*graph.inner);
           })
      .def("untrace", [](GraphRef& graph, Tensor& tensor) {
        AXON_DCHECK(tensor.inst_id().isValid(),
                    "Passed tensor must have a valid inst_id.");
        tensor.untrace();
      });

  m.def("_set_current_graph",
        [](GraphRef graph) -> void { current_graph = graph.inner; });
}

struct BroadcastInfo {
  llvm::SmallVector<insts::ExpandDims::Mapping> expand_dim_mappings;
  llvm::SmallVector<int64_t> unsqueezed_shape;
};

static auto tryGetBroadcastInfo(llvm::ArrayRef<int64_t> source_shape,
                                llvm::ArrayRef<int64_t> target_shape)
    -> std::optional<BroadcastInfo> {
  auto source_rank = static_cast<int64_t>(source_shape.size());
  auto target_rank = static_cast<int64_t>(target_shape.size());

  AXON_DCHECK(
      source_rank < target_rank,
      "Expected the source shape to have lower rank than the target shape");

  BroadcastInfo broadcast_info;

  for (auto i = target_rank - 1; i >= 0; i -= 1) {
    // Calculate corresponding source index (right-aligned)
    auto source_index = i - (target_rank - source_rank);
    auto source_dim = source_index >= 0 ? source_shape[source_index] : 1;
    auto target_dim = target_shape[i];

    broadcast_info.unsqueezed_shape.push_back(source_dim);

    if (source_dim == 1) {
      broadcast_info.expand_dim_mappings.push_back(
          {.dim = i, .scale = target_dim});
      continue;
    }

    // if `source_dim` and `target_dim` are not equal and `source_dim` is not
    // equal to 1 then we cannot broadcast `source_shape` into the
    // `target_shape`.
    if (source_dim != target_dim) {
      return std::nullopt;
    }
  }

  std::reverse(broadcast_info.unsqueezed_shape.begin(),
               broadcast_info.unsqueezed_shape.end());

  return broadcast_info;
}

static auto performBroadcasting(Graph& graph, InstId source_id,
                                llvm::ArrayRef<int64_t> source_shape,
                                llvm::ArrayRef<int64_t> target_shape)
    -> std::optional<InstId> {
  auto broadcast_info = tryGetBroadcastInfo(source_shape, target_shape);
  if (!broadcast_info) {
    return std::nullopt;
  }

  auto reshaped_id = graph.createOp(
      insts::Reshape(source_id, broadcast_info->unsqueezed_shape));
  return graph.createOp(
      insts::ExpandDims(reshaped_id, broadcast_info->expand_dim_mappings));
}

template <typename ElementWiseInst>
static auto performElementWiseOperation(Graph& graph, const Tensor& lhs,
                                        const Tensor& rhs) -> Tensor {
  auto lhs_id = lhs.inst_id();
  auto rhs_id = rhs.inst_id();

  auto lhs_shape = graph.getShape(lhs_id);
  auto rhs_shape = graph.getShape(rhs_id);

  // try broadcasting lhs to have the same shape as rhs
  if (lhs_shape.size() < rhs_shape.size()) {
    auto new_lhs_id = performBroadcasting(graph, lhs_id, lhs_shape, rhs_shape);
    if (!new_lhs_id) {
      throw std::runtime_error(
          std::format("Failed to broadcast {} into {}", lhs_shape, rhs_shape));
    }
    lhs_id = *new_lhs_id;
  }

  // try broadcasting lhs to have the same shape as rhs
  if (lhs_shape.size() > rhs_shape.size()) {
    auto new_rhs_id = performBroadcasting(graph, rhs_id, rhs_shape, lhs_shape);
    if (!new_rhs_id) {
      throw std::runtime_error(
          std::format("Failed to broadcast {} into {}", lhs_shape, rhs_shape));
    }
    rhs_id = *new_rhs_id;
  }

  auto inst_id = graph.createOp(ElementWiseInst(lhs_id, rhs_id));
  return {inst_id};
}

static auto performMatMul(Graph& graph, const Tensor& lhs, const Tensor& rhs)
    -> Tensor {
  static auto is_valid_matmul = [](llvm::ArrayRef<int64_t> lhs_shape,
                                   llvm::ArrayRef<int64_t> rhs_shape) {
    AXON_DCHECK(lhs_shape.size() == rhs_shape.size(),
                "At this point lhs and rhs must have the same rank");

    if (lhs_shape.size() == 3) {
      return lhs_shape[2] == rhs_shape[1];
    }

    if (lhs_shape.size() == 2) {
      return lhs_shape[1] == rhs_shape[0];
    }
    std::unreachable();
  };

  llvm::ArrayRef<int64_t> lhs_shape = lhs.shape();
  llvm::ArrayRef<int64_t> rhs_shape = rhs.shape();

  auto lhs_id = lhs.inst_id();
  auto rhs_id = rhs.inst_id();

  if (lhs_shape.size() > 3 || rhs_shape.size() > 3) {
    throw std::runtime_error(
        "Attempted to multiply tensors with more than rank of 3.");
  }

  if (lhs_shape.size() < rhs_shape.size()) {
    if (lhs_shape.size() == 2 && rhs_shape.size() == 3) {
      llvm::SmallVector<int64_t> target_shape(lhs_shape);
      target_shape.insert(target_shape.begin(), rhs_shape[0]);

      if (not is_valid_matmul(target_shape, rhs_shape)) {
        throw std::runtime_error(
            std::format("Cannot perform matrix multiplication on tensors with "
                        "shape {} and {}",
                        lhs_shape, rhs_shape));
      }

      lhs_id = *performBroadcasting(graph, lhs_id, lhs_shape, target_shape);
    }
  }
  if (lhs_shape.size() > rhs_shape.size()) {
    if (lhs_shape.size() == 3 && rhs_shape.size() == 2) {
      llvm::SmallVector<int64_t> target_shape(rhs_shape);
      target_shape.insert(target_shape.begin(), lhs_shape[0]);

      if (not is_valid_matmul(lhs_shape, target_shape)) {
        throw std::runtime_error(
            std::format("Cannot perform matrix multiplication on tensors with "
                        "shape {} and {}",
                        lhs_shape, rhs_shape));
      }

      rhs_id = *performBroadcasting(graph, rhs_id, rhs_shape, target_shape);
    }
  }

  return graph.createOp(insts::MatMul(lhs_id, rhs_id));
}

static auto buildTensorBindings(nb::module_& m) -> void {
  nb::class_<Tensor>(m, "Tensor")
      .def_prop_ro("shape",
                   [](const Tensor& self) -> nb::tuple {
                     nb::list temp;
                     for (const auto& item : self.shape()) {
                       temp.append(nb::cast(item));
                     }
                     return nb::tuple(temp);
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
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke add since there is no active "
                   "graph.");
             }
             return performElementWiseOperation<insts::Add>(*graph, lhs, rhs);
           })
      .def("__sub__",
           [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke sub since there is no active "
                   "graph.");
             }
             return performElementWiseOperation<insts::Sub>(*graph, lhs, rhs);
           })
      .def("__mul__",
           [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke mul since there is no active "
                   "graph.");
             }
             return performElementWiseOperation<insts::Mul>(*graph, lhs, rhs);
           })
      .def("__mul__",
           [](const Tensor& self, double scalar) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke mul since there is no active "
                   "graph.");
             }
             return graph->createOp(insts::ScalarMul(self.inst_id(), scalar));
           })
      .def("__rmul__",
           [](const Tensor& self, double scalar) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke mul since there is no active "
                   "graph.");
             }
             return graph->createOp(insts::ScalarMul(self.inst_id(), scalar));
           })
      .def("__neg__",
           [](const Tensor& self) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke neg since there is no active "
                   "graph.");
             }
             return graph->createOp(insts::Neg(self.inst_id()));
           })
      .def("__matmul__",
           [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke matmul since there is no active "
                   "graph.");
             }
             return performMatMul(*graph, lhs, rhs);
           })
      .def("backward",
           [](const Tensor& self) -> void {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke backpropagation since there is no active "
                   "graph.");
             }
             axon::backward(*graph, self.inst_id());
           })
      .def_prop_ro("grad", [](const Tensor& self) -> std::shared_ptr<Tensor> {
        return self.grad();
      });

  m.def(
      "_create_tensor",
      [](nb::ndarray<>& array, bool requires_grad,
         DataType::InternalType data_type) -> Tensor {
        auto tensor =
            Tensor(createStoragefromNanobind(array, data_type), requires_grad);
        return tensor;
      },
      nb::rv_policy::move);

  m.def(
      "_create_filled",
      [](nb::tuple shape_object, nb::object fill_object, bool requires_grad,
         DataType::InternalType data_type) -> Tensor {
        llvm::SmallVector<int64_t> shape;
        for (auto dim_object : shape_object) {
          auto dim = nb::cast<int>(dim_object);
          shape.push_back(dim);
        }

        if (data_type == DataType::Float32) {
          auto fill = nb::cast<float>(fill_object);
          auto storage = Storage::createFilled(shape, fill, data_type);
          return {std::move(storage), requires_grad};
        } else if (data_type == DataType::Float64) {
          auto fill = nb::cast<double>(fill_object);
          auto storage = Storage::createFilled(shape, fill, data_type);
          return {std::move(storage), requires_grad};
        }

        throw std::runtime_error("Unsupported DataType");
      },
      nb::rv_policy::move);

  m.def(
      "_create_randn",
      [](nb::tuple shape_object, bool requires_grad,
         DataType::InternalType data_type) -> Tensor {
        llvm::SmallVector<int64_t> shape;
        for (auto dim_object : shape_object) {
          auto dim = nb::cast<int>(dim_object);
          shape.push_back(dim);
        }

        auto storage = Storage::createDistributed(
            shape, /*mean=*/0, /*standard_deviation=*/1.0, data_type);
        return {std::move(storage), requires_grad};
      },
      nb::rv_policy::move);
}

NB_MODULE(_core, m) {
  nb::enum_<DataType::InternalType>(m, "dtype")
      .value("float32", DataType::Float32)
      .value("float64", DataType::Float64)
      .export_values();

  nb::enum_<LoweringLevel>(m, "LoweringLevel")
      .value("Axon", LoweringLevel::Axon)
      .value("Standard", LoweringLevel::Standard)
      .value("Linalg", LoweringLevel::Linalg)
      .value("Loops", LoweringLevel::Loops)
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
