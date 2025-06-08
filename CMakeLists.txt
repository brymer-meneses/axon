cmake_minimum_required(VERSION 3.22)
project(axon)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/axon")

include(FetchContent)

find_package(LLVM 20.1 REQUIRED CONFIG)
find_package(MLIR 20.1 REQUIRED CONFIG)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)


FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind
  GIT_TAG        v2.7.0
  GIT_SUBMODULES_RECURSE ON
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(_axon_impl src/axon/axon.cpp)
install(TARGETS _axon_impl LIBRARY DESTINATION axon)
target_compile_features(_axon_impl PUBLIC cxx_std_23)

set_target_properties(_axon_impl
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/axon"
)
target_link_libraries(_axon_impl PUBLIC
    ${llvm_libs}
    Python::Module
)
