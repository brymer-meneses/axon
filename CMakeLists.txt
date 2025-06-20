cmake_minimum_required(VERSION 3.22)

project(axon)

include(FetchContent)
include(cmake/xtensor.cmake)
include(cmake/nanobind.cmake)

find_package(LLVM 20.1 REQUIRED CONFIG)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs support core irreader)
include_directories(${LLVM_INCLUDE_DIRS})

nanobind_build_library(nanobind SHARED)
nanobind_add_module(_cpp)
target_sources(_cpp 
  PUBLIC FILE_SET CXX_MODULES FILES
  src/cpp/axon.cpp
  src/cpp/tensor.cpp
  src/cpp/convert.cpp
  src/cpp/global_context.cpp
  src/cpp/storage.cpp
  src/cpp/tensor_metadata.cpp
  src/cpp/ids.cpp
)
target_compile_features(_cpp PUBLIC cxx_std_23)
target_link_libraries(_cpp PRIVATE nanobind xtensor)
target_include_directories(_cpp PUBLIC ${xtensor_INCLUDE_DIRS})

set_target_properties(_cpp
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/axon"
)
target_link_libraries(_cpp PUBLIC
    Python::Module
    ${llvm_libs}
)
