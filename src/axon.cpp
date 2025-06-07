#include <nanobind/nanobind.h>
#include <print>

auto add(int a, int b) -> int {
  std::print("Hello world\n");
  return a + b;
}

NB_MODULE(my_ext, m) { m.def("add", &add); }
