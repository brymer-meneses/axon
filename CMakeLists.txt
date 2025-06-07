cmake_minimum_required(VERSION 3.22)
project(axon)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind
  GIT_TAG        v2.7.0
  GIT_SUBMODULES_RECURSE ON
)
FetchContent_MakeAvailable(nanobind)


nanobind_add_module(axon src/axon.cpp)
target_compile_features(axon PUBLIC cxx_std_23)
