module;

#include <vector>

#include "nanobind/ndarray.h"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"

export module axon.convert;

namespace axon {

namespace nb = nanobind;

export template <typename T>
auto ConvertNDArrayToXArray(nb::ndarray<nb::numpy, T> nb_array)
    -> xt::xarray<T> {
  std::vector<size_t> shape;
  for (auto i = 0; i < nb_array.ndim(); ++i) {
    shape.push_back(nb_array.shape(i));
  }

  auto xt_view = xt::adapt(static_cast<const T*>(nb_array.data()), shape);
  return xt::xarray<T>(xt_view);
}
}  // namespace axon
