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

template <Numeric T>
static auto checkIsWithinTolerance(T* lhs, llvm::ArrayRef<i64> lhs_strides,
                                   T* rhs, llvm::ArrayRef<i64> rhs_strides,
                                   llvm::ArrayRef<i64> shape, f64 tolerance)
    -> void {
  auto rank = static_cast<i64>(shape.size());
  if (rank == 0) {
    if (std::fabs(*lhs - *rhs) > tolerance) {
      throw nb::value_error(
          std::format("Mismatch expected {} got {}", *rhs, *lhs).c_str());
    }
    return;
  }

  llvm::SmallVector<i64> idx(rank, 0);
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
    auto lhs_off = 0;
    auto rhs_off = 0;

    for (i64 d = 0; d < rank; ++d) {
      lhs_off += idx[d] * lhs_strides[d];
      rhs_off += idx[d] * rhs_strides[d];
    }
    auto a = lhs[lhs_off];
    auto b = rhs[rhs_off];
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

NB_MODULE(_core, m) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

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
             if (nb::isinstance<nb::ndarray<nb::c_contig, nb::ro>>(object)) {
               auto ndarray =
                   nb::cast<nb::ndarray<nb::c_contig, nb::ro>>(object);
               auto ndarray_data_type = DataType::fromDlPack(ndarray.dtype());
               if (ndarray_data_type != data_type) {
                 throw nb::value_error("Does not match the received data type");
               }

               auto storage = Storage::fromDlPack(ndarray, data_type);
               return std::make_shared<Tensor>(std::move(storage),
                                               requires_grad);
             }

             if (nb::isinstance<nb::list>(object)) {
               auto storage =
                   Storage::fromPython(nb::cast<nb::list>(object), data_type);
               return std::make_shared<Tensor>(std::move(storage),
                                               requires_grad);
             }
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
      .def("__rmul__", &performScalarMul<f32>)

      .def("pow", &performPow<f32>, nb::arg("exponent"))
      .def("__pow__", &performPow<f32>)

      .def("__matmul__", &performMatMul)

      .def("softmax", &performSoftmax, nb::arg("dim"))

      .def("relu", &performUnaryInst<insts::Relu>)

      .def("accumulate", &performAccumulate, nb::arg("value"))

      .def("sum", &performReduceInst<insts::Sum>, nb::arg("dim") = std::nullopt,
           nb::arg("keepdims") = false)

      .def("mean", &performReduceInst<insts::Mean>,
           nb::arg("dim") = std::nullopt, nb::arg("keepdims") = false)

      .def_static(
          "ones",
          [](nb::tuple shape_tuple, bool requires_grad,
             DataType::InternalType data_type) {
            auto shape = convertTupleToVector(shape_tuple);
            auto storage = Storage::createFilled(shape, 1.0, data_type);
            return std::make_shared<Tensor>(std::move(storage), requires_grad);
          },
          nb::arg("shape"), nb::arg("requires_grad") = false,
          nb::arg("dtype") = DataType::Float32)

      .def_static(
          "zeros",
          [](nb::tuple shape_tuple, bool requires_grad,
             DataType::InternalType data_type) {
            auto shape = convertTupleToVector(shape_tuple);
            auto storage = Storage::createFilled(shape, 0.0, data_type);
            return std::make_shared<Tensor>(std::move(storage), requires_grad);
          },
          nb::arg("shape"), nb::arg("requires_grad") = false,
          nb::arg("dtype") = DataType::Float32)

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
          nb::arg("dtype") = DataType::Float32);

  m.def(
      "assert_are_close",
      [](std::shared_ptr<Tensor> tensor,
         nb::ndarray<nb::ro, nb::c_contig>& array, f64 tolerance) {
        auto array_strides =
            llvm::ArrayRef<i64>(array.stride_ptr(), array.ndim());
        auto array_shape = llvm::ArrayRef<i64>(array.shape_ptr(), array.ndim());

        if (!tensor->isEvaluated()) {
          tensor->evaluate();
        }

        if (tensor->shape() != array_shape) {
          throw nb::value_error("Do not have the same shape");
        }

        auto data_type = DataType::fromDlPack(array.dtype());
        if (data_type != tensor->data_type()) {
          throw nb::value_error("Do not have the same data type");
        }

        auto shape = tensor->shape();
        auto lhs_strides = tensor->storage()->strides();
        auto rhs_strides = array_strides;
        switch (tensor->data_type().kind()) {
          case DataType::Float32: {
            auto tensor_ptr = tensor->storage()->as<f32>();
            auto data_ptr = reinterpret_cast<const f32*>(array.data());
            checkIsWithinTolerance<const f32>(tensor_ptr, lhs_strides, data_ptr,
                                              rhs_strides, shape, tolerance);
            break;
          }
          case DataType::Float64: {
            auto tensor_ptr = tensor->storage()->as<f64>();
            auto data_ptr = reinterpret_cast<const f64*>(array.data());
            checkIsWithinTolerance<const f64>(tensor_ptr, lhs_strides, data_ptr,
                                              rhs_strides, shape, tolerance);
            break;
          }
        }
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
