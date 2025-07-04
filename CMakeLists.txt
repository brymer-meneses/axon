cmake_minimum_required(VERSION 4.0.2)
project(axon)

include(cmake/dependencies.cmake)

add_library(libaxon)
target_sources(libaxon 
  PUBLIC FILE_SET CXX_MODULES FILES
  src/base/value_store.cpp
  src/base/index_base.cpp

  src/core/module.cpp
  src/core/inst.cpp
  src/core/ids.cpp
  src/core/eager.cpp

  src/mlir/codegen.cpp
)
target_compile_features(libaxon PUBLIC cxx_std_23)
target_include_directories(libaxon
  SYSTEM PRIVATE 
  ${xtensor_INCLUDE_DIRS} 
  ${NANOBIND_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(libaxon
  PRIVATE
  xtensor 
  xtensor-blas
  ${llvm_libs}
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRArithDialect
)
if (MSVC)
  target_compile_options(libaxon PRIVATE /W4 /WX)
else()
  target_compile_options(libaxon PRIVATE -Wall -Wextra -Werror -Wpedantic)
endif()

nanobind_build_library(nanobind SHARED)
add_library(_cpp)
target_sources(_cpp 
  PUBLIC
  src/python/bindings.cpp
)
set_target_properties(_cpp
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python/axon"
)
target_link_libraries(_cpp PUBLIC libaxon nanobind)

# Mainly for debugging for now.
add_executable(inspector)
target_sources(inspector 
  PUBLIC
  src/inspector.cpp
)
target_compile_features(inspector PUBLIC cxx_std_23)
target_link_libraries(inspector 
  PUBLIC 
  libaxon
)
