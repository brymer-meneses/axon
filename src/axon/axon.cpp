#include <Python.h>
#include <nanobind/nanobind.h>

#include <print>

namespace nb = nanobind;

auto func(nb::object object) {
  if (not object.is_valid()) {
    throw std::runtime_error("Passed object is null.");
  }

  if (not PyFunction_Check(object.ptr())) {
    throw std::invalid_argument("Passed object is not a function.");
  }

  std::println("Hello world!");
}

NB_MODULE(_axon_impl, m) { m.def("func", &func); }
