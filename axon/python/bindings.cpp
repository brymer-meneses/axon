#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "axon/base/macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "nanobind/intrusive/counter.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/unique_ptr.h"
#include "nanobind/stl/vector.h"

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

static auto checkIsWithinTolerance(const Storage* lhs, const Storage* rhs,
                                   f64 tolerance) -> void {
  auto shape = lhs->shape();
  if (rhs->shape() != shape) {
    throw nb::value_error("Do not have the same shape");
  }
  if (lhs->data_type() != rhs->data_type()) {
    throw nb::value_error("Do not have the same data type");
  }

  auto rank = static_cast<i64>(shape.size());
  if (rank == 0) {
    auto a = lhs->getElementAt({}).getAs<f64>();
    auto b = rhs->getElementAt({}).getAs<f64>();
    if (std::fabs(a - b) > tolerance) {
      throw nb::value_error(
          std::format("Mismatch expected {} got {}", b, a).c_str());
    }
    return;
  }

  llvm::SmallVector<i64, 8> idx(rank, 0);
  auto advance = [&]() -> bool {
    for (i64 d = rank - 1; d >= 0; --d) {
      idx[d] += 1;
      if (idx[d] < shape[d]) {
        return true;
      }
      idx[d] = 0;
    }
    return false;
  };

  while (true) {
    auto a = lhs->getElementAt(idx).getAs<f64>();
    auto b = rhs->getElementAt(idx).getAs<f64>();
    if (std::fabs(a - b) > tolerance) {
      throw nb::value_error(std::format("Mismatch at {}: lhs={} rhs={} tol={}",
                                        idx, a, b, tolerance)
                                .c_str());
    }
    if (!advance()) {
      break;
    }
  }
}

static auto createFillLike(Scalar fill_value, llvm::ArrayRef<i64> shape,
                           bool requires_grad) -> std::shared_ptr<Tensor> {
  auto cpu = CpuStorage::createFilled(fill_value, shape);
  return std::make_shared<Tensor>(Storage(std::move(cpu)), requires_grad);
}

static auto createScalarFromPythonObject(nb::handle value,
                                         DataType::InternalType data_type)
    -> Scalar {
  try {
    switch (data_type) {
      case DataType::Float32:
        return Scalar(static_cast<f32>(nb::cast<double>(value)));
      case DataType::Float64:
        return Scalar(nb::cast<double>(value));
      case DataType::Int1:
        return Scalar(static_cast<bool>(nb::cast<bool>(value)));
      case DataType::Int32: {
        auto casted = nb::cast<i64>(value);
        if (casted < std::numeric_limits<i32>::min() ||
            casted > std::numeric_limits<i32>::max()) {
          throw nb::value_error(
              "value does not fit within the requested dtype");
        }
        return Scalar(static_cast<i32>(casted));
      }
      case DataType::Int64:
        return Scalar(nb::cast<i64>(value));
    }
  } catch (const nb::cast_error&) {
    throw nb::value_error(
        "value has incompatible type for the requested dtype");
  }
  AXON_UNREACHABLE("Unhandled dtype when creating scalar");
}

NB_MODULE(_core, m) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  nb::enum_<DataType::InternalType>(m, "dtype")
      .value("float32", DataType::Float32)
      .value("float64", DataType::Float64)
      .value("bool", DataType::Int1)
      .value("int32", DataType::Int32)
      .value("int64", DataType::Int64)
      .export_values();

  nb::enum_<LoweringLevel>(m, "LoweringLevel")
      .value("Axon", LoweringLevel::Axon)
      .value("Standard", LoweringLevel::Standard)
      .value("Linalg", LoweringLevel::Linalg)
      .value("Loops", LoweringLevel::Loops)
      .value("LLVM", LoweringLevel::LLVM)
      .export_values();

  m.def(
      "inspect_ir",
      [](std::shared_ptr<Tensor> tensor, LoweringLevel level) {
        mlir::MLIRContext context(createDialectRegistry());
        context.loadAllAvailableDialects();

        mlir::OpBuilder builder(&context);
        mlir::PassManager pm(&context);

        auto session = tensor->session();
        if (!session) {
          throw nb::value_error("Failed to inspect a non-lazy tensor.");
        }

        session->setReturned(tensor);

        auto module = codegenGraph(session->graph(), builder);
        axon::buildLlvmLoweringPipeline(pm, level);

        auto lowering_result = pm.run(module);
        if (lowering_result.failed()) {
          throw std::runtime_error("Failed to compile graph.");
        }

        session->setReturnedToNone();

        module.print(llvm::outs());
      },
      nb::arg("tensor"), nb::arg("level"));

  nb::class_<Tensor>(m, "Tensor")
      .def(nb::new_([](nb::object object, bool requires_grad,
                       DataType::InternalType data_type) {
             if (nb::isinstance<nb::ndarray<nb::c_contig>>(object)) {
               auto ndarray = nb::cast<nb::ndarray<nb::c_contig>>(object);
               auto storage = WritableDlPackStorage(ndarray);
               return std::make_shared<Tensor>(Storage(std::move(storage)),
                                               requires_grad);
             }

             if (nb::isinstance<nb::list>(object)) {
               auto cpu = CpuStorage::fromPythonList(nb::cast<nb::list>(object),
                                                     data_type);
               return std::make_shared<Tensor>(Storage(std::move(cpu)),
                                               requires_grad);
             }

             AXON_UNREACHABLE("unreachable");
           }),
           nb::arg("data"), nb::arg("requires_grad") = false,
           nb::arg("dtype") = DataType::Float32)

      .def_prop_ro(
          "shape",
          [](Tensor& self) { return convertVectorToTuple(self.shape()); })

      .def_prop_ro("requires_grad", &Tensor::requiresGrad)
      .def_prop_ro("grad", &Tensor::grad)

      .def_prop_ro("is_evaluated", &Tensor::isEvaluated)

      .def("zero_grad", &Tensor::zeroGrad)

      .def("evaluate", &Tensor::evaluate)

      .def("backward", &Tensor::backward, nb::arg("grad") = nullptr)

      .def("__repr__", &Tensor::asString)

      .def("__neg__", &performUnaryInst<insts::Neg>)

      .def("__eq__", &performComparison<insts::Compare::Predicate::Equal>)
      .def("__ne__", &performComparison<insts::Compare::Predicate::NotEqual>)
      .def("__le__", &performComparison<insts::Compare::Predicate::LessEq>)
      .def("__lt__", &performComparison<insts::Compare::Predicate::Less>)
      .def("__ge__", &performComparison<insts::Compare::Predicate::GreaterEq>)
      .def("__gt__", &performComparison<insts::Compare::Predicate::Greater>)

      .def("__add__", &performBinaryElementWiseOperation<insts::Add>)
      .def("__mul__", &performBinaryElementWiseOperation<insts::Mul>)
      .def("__sub__", &performBinaryElementWiseOperation<insts::Sub>)

      .def("__mul__", &performScalarMul<f32>)
      .def("__mul__", &performScalarMul<f64>)
      .def("__mul__", &performScalarMul<i32>)
      .def("__mul__", &performScalarMul<i64>)
      .def("__rmul__", &performScalarMul<f32>)
      .def("__rmul__", &performScalarMul<f64>)
      .def("__rmul__", &performScalarMul<i32>)
      .def("__rmul__", &performScalarMul<i64>)

      .def("pow", &performPow<f32>, nb::arg("exponent"))
      .def("pow", &performPow<f64>, nb::arg("exponent"))
      .def("__pow__", &performPow<f32>)
      .def("__pow__", &performPow<f64>)

      .def("__matmul__", &performMatMul)

      .def("softmax", &performSoftmax, nb::arg("dim"))

      .def("relu", &performUnaryInst<insts::Relu>)
      .def("log", &performUnaryInst<insts::Log>)

      .def("accumulate", &performAccumulate, nb::arg("value"))

      .def("sum", &performReduceInst<insts::Sum>, nb::arg("dim") = std::nullopt,
           nb::arg("keepdims") = false)

      .def("mean", &performReduceInst<insts::Mean>,
           nb::arg("dim") = std::nullopt, nb::arg("keepdims") = false)

      .def("item",
           [](std::shared_ptr<Tensor> self) -> nb::object {
             if (self->rank() != 0) {
               throw nb::value_error(
                   std::format("Cannot perform item on a tensor with a "
                               "non-scalar shape of "
                               "{}.",
                               self->shape())
                       .c_str());
             }
             if (!self->isEvaluated()) {
               self->evaluate();
             }
             auto dtype = self->data_type();
             switch (dtype.kind()) {
               case DataType::Float32:
                 return nb::cast(*self->storage()->as<f32>());
               case DataType::Float64:
                 return nb::cast(*self->storage()->as<f64>());
               case DataType::Int1:
                 return nb::cast(*self->storage()->as<bool>());
               case DataType::Int32:
                 return nb::cast(*self->storage()->as<i32>());
               case DataType::Int64:
                 return nb::cast(*self->storage()->as<i64>());
             }
             AXON_UNREACHABLE("Unsupported dtype for item()");
           })

      .def_static(
          "fill_like",
          [](nb::object value, nb::tuple shape_tuple, bool requires_grad,
             DataType::InternalType data_type) {
            auto shape = convertTupleToVector(shape_tuple);
            auto scalar = createScalarFromPythonObject(value, data_type);
            return createFillLike(scalar, shape, requires_grad);
          },
          nb::arg("value"), nb::arg("shape"), nb::arg("requires_grad") = false,
          nb::arg("dtype") = DataType::Float32)

      .def_static(
          "ones",
          [](nb::tuple shape_tuple, bool requires_grad,
             DataType::InternalType data_type) {
            auto shape = convertTupleToVector(shape_tuple);
            return createFillLike(Scalar(1.0).cast(data_type), shape,
                                  requires_grad);
          },
          nb::arg("shape"), nb::arg("requires_grad") = false,
          nb::arg("dtype") = DataType::Float32)

      .def_static(
          "zeros",
          [](nb::tuple shape_tuple, bool requires_grad,
             DataType::InternalType data_type) {
            auto shape = convertTupleToVector(shape_tuple);
            return createFillLike(Scalar(0.0).cast(data_type), shape,
                                  requires_grad);
          },
          nb::arg("shape"), nb::arg("requires_grad") = false,
          nb::arg("dtype") = DataType::Float32)

      .def_static(
          "randn",
          [](nb::tuple shape_tuple, bool requires_grad,
             DataType::InternalType data_type) {
            auto shape = convertTupleToVector(shape_tuple);
            if (data_type != DataType::Float32 &&
                data_type != DataType::Float64) {
              throw nb::value_error(
                  "randn is only supported for floating point dtypes");
            }
            auto cpu =
                CpuStorage::createDistributed(shape, 0.0, 1.0, data_type);
            return std::make_shared<Tensor>(Storage(std::move(cpu)),
                                            requires_grad);
          },
          nb::arg("shape"), nb::arg("requires_grad") = false,
          nb::arg("dtype") = DataType::Float32);

  m.def(
      "assert_are_close",
      [](std::shared_ptr<Tensor> tensor,
         nb::ndarray<nb::c_contig, nb::ro>& array, f64 tolerance) {
        if (!tensor->isEvaluated()) {
          tensor->evaluate();
        }
        auto rhs = Storage(ReadOnlyDlPackStorage(array));

        checkIsWithinTolerance(tensor->storage(), &rhs, tolerance);
      },
      nb::arg("tensor"), nb::arg("ndarray"), nb::arg("tolerance") = 1e-5);

  m.def("total_number_of_compiled_functions", []() {
    auto& instance = Runtime::get();
    return instance.getTotalNumberOfCompiledFunctions();
  });

  m.def("set_emit_grad", [](bool value) {
    auto& instance = Runtime::get();
    instance.setEmitGrad(value);
  });
}
