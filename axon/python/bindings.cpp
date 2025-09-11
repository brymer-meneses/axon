#include <cmath>
#include <format>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/OperationSupport.h"
#include "nanobind/intrusive/counter.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/unique_ptr.h"
#include "nanobind/stl/vector.h"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xio.hpp"

import axon.base;
import axon.core;
import axon.python;
import axon.mlir;

namespace nb = nanobind;

using namespace axon;

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
             auto compilation_unit =
                 std::make_unique<CompilationUnit>(*graph.inner);
             if (compilation_unit->compile(level).failed()) {
               throw std::runtime_error("Failed to compile graph");
             }
             return std::move(compilation_unit);
           })
      .def("create_constant",
           [](GraphRef& graph, nb::ndarray<>& array,
              DataType::InternalType data_type) -> Tensor {
             auto data = Storage::fromNanobind(array, data_type);
             auto inst_id = graph.inner->createConstant(std::move(data));
             return Tensor(inst_id);
           })
      .def("trace",
           [](GraphRef& graph, Tensor& tensor) {
             AXON_DCHECK(tensor.hasData(), "Passed tensor must have data.");
             tensor.trace(*graph.inner);
           })
      .def("untrace",
           [](GraphRef& graph, Tensor& tensor) {
             AXON_DCHECK(tensor.inst_id().isValid(),
                         "Passed tensor must have a valid inst_id.");
             tensor.untrace();
           })
      .def("set_returned", [](GraphRef& graph, Tensor& tensor) {
        AXON_DCHECK(tensor.inst_id().isValid(),
                    "Passed tensor must have a valid inst_id.");
        graph.inner->setReturned(tensor.inst_id());
      });

  m.def("_set_current_graph",
        [](GraphRef graph) -> void { current_graph = graph.inner; });
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

      .def("transpose",
           [](const Tensor& self, i64 from, i64 to) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke add since there is no active "
                   "graph.");
             }
             auto inst_id =
                 graph->createOp(insts::Transpose(self.inst_id(), from, to));
             return Tensor(inst_id);
           })

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
           [](const Tensor& self, f32 scalar) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke mul since there is no active "
                   "graph.");
             }
             return Tensor(graph->createOp(
                 insts::ScalarMul(self.inst_id(), Scalar(scalar))));
           })

      // FIXME: changing this to a f64 results to invalid folding due to APFloat
      // having different semantics.
      .def("__rmul__",
           [](const Tensor& self, f32 scalar) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke mul since there is no active "
                   "graph.");
             }
             return Tensor(graph->createOp(
                 insts::ScalarMul(self.inst_id(), Scalar(scalar))));
           })

      .def("__neg__",
           [](const Tensor& self) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke neg since there is no active "
                   "graph.");
             }
             return Tensor(graph->createOp(insts::Neg(self.inst_id())));
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
      .def("sum",
           [](const Tensor& self, i32 axis) -> Tensor {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke sum since there is no active "
                   "graph.");
             }
             return Tensor(graph->createOp(insts::Sum(self.inst_id(), axis)));
           })
      .def("backward",
           [](const Tensor& self, const Tensor& grad) -> void {
             auto graph = current_graph.lock();
             if (!graph) {
               throw std::runtime_error(
                   "Failed to invoke backpropagation since there is no active "
                   "graph.");
             }
             axon::backward(*graph, self.inst_id(), grad.inst_id());
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
            Tensor(Storage::fromNanobind(array, data_type), requires_grad);
        return tensor;
      },
      nb::rv_policy::move);

  m.def(
      "_create_filled",
      [](nb::tuple shape_object, nb::object fill_object, bool requires_grad,
         DataType::InternalType data_type) -> Tensor {
        llvm::SmallVector<i64> shape;
        for (auto dim_object : shape_object) {
          auto dim = nb::cast<int>(dim_object);
          shape.push_back(dim);
        }

        if (data_type == DataType::Float32) {
          auto fill = nb::cast<float>(fill_object);
          auto storage = Storage::createFilled(shape, fill, data_type);
          return {std::move(storage), requires_grad};
        } else if (data_type == DataType::Float64) {
          auto fill = nb::cast<f64>(fill_object);
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
        llvm::SmallVector<i64> shape;
        for (auto dim_object : shape_object) {
          auto dim = nb::cast<i32>(dim_object);
          shape.push_back(dim);
        }

        auto storage = Storage::createDistributed(
            shape, /*mean=*/0, /*standard_deviation=*/1.0, data_type);
        return {std::move(storage), requires_grad};
      },
      nb::rv_policy::move);
}

static auto hasDataType(DataType data_type, nb::dlpack::dtype dtype) -> bool {
  if (data_type == DataType::Float32) {
    return dtype == nb::dlpack::dtype{
                        .code = static_cast<u8>(nb::dlpack::dtype_code::Float),
                        .bits = 32,
                        .lanes = 1};
  }

  if (data_type == DataType::Float64) {
    return dtype == nb::dlpack::dtype{
                        .code = static_cast<u8>(nb::dlpack::dtype_code::Float),
                        .bits = 64,
                        .lanes = 1};
  }

  AXON_UNREACHABLE("TODO");
}

template <typename T>
static auto checkIsWithinTolerance(T* lhs, T* rhs, i64 total_elements,
                                   T tolerance) -> bool {
  for (i64 i = 0; i < total_elements; ++i) {
    if (std::fabs(lhs[i] - rhs[i]) > tolerance) {
      return false;
    }
  }
  return true;
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
      .def(
          "execute",
          [](CompilationUnit& self,
             std::vector<std::shared_ptr<Tensor>> tensors)
              -> std::optional<Tensor> {
            return self.execute(std::move(tensors));
          },
          nb::rv_policy::move);

  buildTensorBindings(m);
  buildGraphBindings(m);

  m.def("_is_equal", [](const Tensor& tensor, nb::ndarray<>& array) -> bool {
    auto array_strides = llvm::ArrayRef<i64>(array.stride_ptr(), array.ndim());
    auto array_shape = llvm::ArrayRef<i64>(array.shape_ptr(), array.ndim());
    if (not tensor.hasData()) {
      return false;
    }
    if (tensor.data()->shape() != array_shape) {
      return false;
    }
    if (tensor.data()->strides() != array_strides) {
      return false;
    }
    if (!hasDataType(tensor.data()->data_type(), array.dtype())) {
      return false;
    }
    auto size = array.size();
    switch (tensor.data()->data_type().kind()) {
      case DataType::Float32: {
        auto tensor_ptr = tensor.data()->as<f32>();
        auto data_ptr = reinterpret_cast<f32*>(array.data());
        return checkIsWithinTolerance<const f32>(tensor_ptr, data_ptr, size,
                                                 0.0);
      }
      case DataType::Float64: {
        auto tensor_ptr = tensor.data()->as<f64>();
        auto data_ptr = reinterpret_cast<f64*>(array.data());
        return checkIsWithinTolerance<const f64>(tensor_ptr, data_ptr, size,
                                                 0.0);
      }
    }
  });

  m.def(
      "_is_close",
      [](const Tensor& tensor, nb::ndarray<>& array, float tolerance) -> bool {
        auto array_strides =
            llvm::ArrayRef<i64>(array.stride_ptr(), array.ndim());
        auto array_shape = llvm::ArrayRef<i64>(array.shape_ptr(), array.ndim());
        if (not tensor.hasData()) {
          return false;
        }
        if (tensor.data()->shape() != array_shape) {
          std::println("Do not have the same shape");
          return false;
        }
        if (tensor.data()->strides() != array_strides) {
          std::println("Do not have the same strides");
          return false;
        }
        if (!hasDataType(tensor.data()->data_type(), array.dtype())) {
          std::println("Do not have the same data type");
          return false;
        }

        auto size = array.size();
        switch (tensor.data()->data_type().kind()) {
          case DataType::Float32: {
            auto tensor_ptr = tensor.data()->as<f32>();
            auto data_ptr = reinterpret_cast<f32*>(array.data());
            return checkIsWithinTolerance<const f32>(tensor_ptr, data_ptr, size,
                                                     tolerance);
          }
          case DataType::Float64: {
            auto tensor_ptr = tensor.data()->as<f64>();
            auto data_ptr = reinterpret_cast<f64*>(array.data());
            return checkIsWithinTolerance<const f64>(tensor_ptr, data_ptr, size,
                                                     tolerance);
          }
        }
      });
}
