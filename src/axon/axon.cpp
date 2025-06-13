module;

#include <nanobind/nanobind.h>

#include <print>

export module axon;
export import axon.op;
export import axon.tensor;

namespace nb = nanobind;

NB_MODULE(_axon_cpp, m) { std::println("Hello world!"); }
