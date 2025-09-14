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
#include "mlir/Transforms/DialectConversion.h"
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

static auto convertTupleToVector(nb::tuple tuple) -> llvm::SmallVector<i64> {
  llvm::SmallVector<i64, 4> shape;

  for (auto dim : tuple) {
    if (!nb::isinstance<i64>(dim)) {
      throw nb::value_error("Expected a tuple of int");
    }
    shape.emplace_back(nb::cast<i64>(dim));
  }

  return shape;
}

static auto convertVectorToTuple(llvm::ArrayRef<i64> shape) -> nb::tuple {
  nb::list list;
  for (auto dim : shape) {
    list.append(dim);
  }
  return nb::tuple(list);
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

  nb::class_<Tensor>(m, "Tensor")
      .def(nb::new_([](nb::object object, bool requires_grad,
                       DataType::InternalType data_type) {
             if (!nb::isinstance<nb::list>(object)) {
               throw nb::value_error("Expected a list");
             }
             auto storage =
                 Storage::fromPython(nb::cast<nb::list>(object), data_type);
             return std::make_shared<Tensor>(std::move(storage), requires_grad);
           }),
           nb::arg("data"), nb::arg("requires_grad") = false,
           nb::arg("dtype") = DataType::Float64)

      .def_prop_ro(
          "shape",
          [](Tensor& self) { return convertVectorToTuple(self.shape()); })

      .def_prop_ro("requires_grad", &Tensor::requiresGrad)

      .def_prop_ro("is_evaluated", &Tensor::isEvaluated)

      .def("backward", &Tensor::backward)

      .def("__repr__", &Tensor::asString)

      .def("__add__", &performBinaryElementWiseOperation<insts::Add>)
      .def("__mul__", &performBinaryElementWiseOperation<insts::Mul>)
      .def("__sub__", &performBinaryElementWiseOperation<insts::Sub>)

      .def_static(
          "ones",
          [](nb::tuple shape_tuple, bool requires_grad,
             DataType::InternalType data_type) {
            auto shape = convertTupleToVector(shape_tuple);
            auto storage = Storage::createFilled(shape, 1.0, data_type);
            return std::make_shared<Tensor>(std::move(storage), requires_grad);
          },
          nb::arg("shape"), nb::arg("requires_grad") = false,
          nb::arg("dtype") = DataType::Float64)

      .def_static(
          "zeros",
          [](nb::tuple shape_tuple, bool requires_grad,
             DataType::InternalType data_type) {
            auto shape = convertTupleToVector(shape_tuple);
            auto storage = Storage::createFilled(shape, 0.0, data_type);
            return std::make_shared<Tensor>(std::move(storage), requires_grad);
          },
          nb::arg("shape"), nb::arg("requires_grad") = false,
          nb::arg("dtype") = DataType::Float64)

      .def_static(
          "randn",
          [](nb::tuple shape_tuple, bool requires_grad,
             DataType::InternalType data_type) {
            auto shape = convertTupleToVector(shape_tuple);
            auto storage =
                Storage::createDistributed(shape, 0.0, 1.0, data_type);
            return std::make_shared<Tensor>(std::move(storage), requires_grad);
          },
          nb::arg("shape"), nb::arg("requires_grad") = false,
          nb::arg("dtype") = DataType::Float64);
}
