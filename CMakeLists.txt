cmake_minimum_required(VERSION 3.22)

project(axon)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(cmake/eigen.cmake)
include(cmake/nanobind.cmake)

find_package(LLVM 20.1 REQUIRED CONFIG)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs support core irreader)
include_directories(${LLVM_INCLUDE_DIRS})

nanobind_build_library(nanobind SHARED)
nanobind_add_module(_axon_cpp)
target_sources(_axon_cpp 
  PUBLIC FILE_SET CXX_MODULES FILES
  src/axon/axon.cpp
  src/axon/tensor.cpp
  src/axon/op.cpp
  src/axon/context.cpp
  src/axon/basic_storage.cpp
)
target_compile_features(_axon_cpp PUBLIC cxx_std_23)
target_link_libraries(_axon_cpp PRIVATE nanobind)
target_include_directories(_axon_cpp PRIVATE ${CMAKE_SOURCE_DIR}/src)

nanobind_opt_size(_axon_cpp)
nanobind_lto(_axon_cpp)
nanobind_set_visibility(_axon_cpp)
nanobind_disable_stack_protector(_axon_cpp)
nanobind_extension(_axon_cpp)
nanobind_link_options(_axon_cpp)

set_target_properties(_axon_cpp
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/axon"
)
target_link_libraries(_axon_cpp PUBLIC
    Python::Module
    ${llvm_libs}
)
