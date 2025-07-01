module;

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xio.hpp"

export module axon;
//
// import axon.graph;
// import axon.tensor;
// import axon.context;
// import axon.inst;

namespace nb = nanobind;

// static auto convert_to_xarray(nb::ndarray<nb::numpy, float> nb_array)
//     -> xt::xarray<float> {
//   std::vector<size_t> shape;
//   for (auto i = 0u; i < nb_array.ndim(); ++i) {
//     shape.push_back(nb_array.shape(i));
//   }
//
//   auto xt_view = xt::adapt(static_cast<const float*>(nb_array.data()),
//   shape); return xt::xarray<float>(xt_view);
// }
//
NB_MODULE(_cpp, _) {}
