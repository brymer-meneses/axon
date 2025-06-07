#include <nanobind/nanobind.h>

auto add(int a, int b) -> int { return a + b; }

NB_MODULE(_axon_impl, m) { m.def("add", &add); }
