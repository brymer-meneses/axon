cmake_minimum_required(VERSION 3.22)
project(axon)

include(cmake/dependencies.cmake)

nanobind_add_module(_cpp)

target_sources(_cpp 
  PUBLIC FILE_SET CXX_MODULES FILES
  src/python/bindings.cpp
  src/python/tensor.cpp

  src/core/graph.cpp
  src/core/inst.cpp
  src/core/ids.cpp
  src/core/eager.cpp

  src/base/value_store.cpp
  src/base/index_map.cpp
  src/base/index_base.cpp
)

target_compile_features(_cpp 
  PRIVATE 
  cxx_std_23
)

if (MSVC)
  target_compile_options(_cpp PRIVATE /W4 /WX)
else()
  target_compile_options(_cpp PRIVATE -Wall -Wextra -Werror)
endif()

target_link_libraries(_cpp 
  PRIVATE 
  xtensor 
  xtensor-blas

  ${llvm_libs}

  MLIRAnalysis
  MLIRFunctionInterfaces
)

target_include_directories(_cpp PUBLIC ${xtensor_INCLUDE_DIRS})

set_target_properties(_cpp
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python/axon"
)
