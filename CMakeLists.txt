cmake_minimum_required(VERSION 4.0.2)
project(axon)

include(cmake/dependencies.cmake)

add_library(libaxon)

target_sources(libaxon 
  PUBLIC FILE_SET CXX_MODULES FILES
    axon/base/storage.cpp
    axon/base/index_base.cpp
    axon/base/base.cpp

    axon/core/core.cpp
    axon/core/inst.cpp
    axon/core/inst_kinds.cpp
    axon/core/inst_rules.cpp
    axon/core/ids.cpp
    axon/core/eager.cpp
    axon/core/module.cpp

    axon/mlir/codegen_module.cpp
    axon/mlir/codegen_inst.cpp
    axon/mlir/mlir.cpp
    axon/mlir/context.cpp
  PRIVATE
    axon/mlir/dialect/dialect.cpp
    axon/mlir/dialect/types.cpp
    axon/mlir/dialect/ops.cpp
)

target_compile_features(libaxon PUBLIC cxx_std_23)
target_include_directories(libaxon
  SYSTEM PRIVATE 
  ${xtensor_INCLUDE_DIRS} 
  ${NANOBIND_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}

  # The generated files by `mlir-tblgen` are stored here
  ${CMAKE_CURRENT_BINARY_DIR}

  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(libaxon
  PRIVATE
  xtensor 
  xtensor-blas
  ${llvm_libs}
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRArithDialect
)

if (MSVC)
  target_compile_options(libaxon PRIVATE /W4 /WX)
else()
  target_compile_options(libaxon PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(LLVM_TARGET_DEFINITIONS axon/mlir/dialect/dialect.td)
mlir_tablegen(generated/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(generated/Dialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(generated/DialectTypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(generated/DialectTypeDefs.cpp.inc -gen-typedef-defs)
mlir_tablegen(generated/DialectOps.h.inc -gen-op-decls)
mlir_tablegen(generated/DialectOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AxonDialectIncGen)
add_dependencies(libaxon AxonDialectIncGen)

nanobind_build_library(nanobind SHARED)
add_library(_cpp)
target_sources(_cpp 
  PUBLIC
  axon/python/bindings.cpp
)
set_target_properties(_cpp
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/axon/python/axon"
)
target_link_libraries(_cpp PUBLIC libaxon nanobind)

# Mainly for debugging for now.
add_executable(inspector)
target_sources(inspector 
  PUBLIC
  axon/inspector.cpp
)
target_compile_features(inspector PUBLIC cxx_std_23)
target_link_libraries(inspector 
  PUBLIC 
  libaxon
)
