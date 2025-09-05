cmake_minimum_required(VERSION 4.0.2)

project(axon)

include(cmake/dependencies.cmake)

add_library(mlir INTERFACE)
target_link_libraries(mlir
  INTERFACE
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRDialect
    MLIRLLVMDialect
    MLIRTensorDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRBufferizationDialect
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRTensorToLinalg
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRArithToLLVM
    MLIRBufferizationPipelines
    MLIRReconcileUnrealizedCasts
    MLIRFuncTransforms
    MLIRExecutionEngine
    MLIRFuncAllExtensions
)


add_library(axon STATIC)

target_sources(axon 
  PUBLIC FILE_SET CXX_MODULES FILES
    axon/base/storage.cpp
    axon/base/index_base.cpp
    axon/base/base.cpp

    axon/core/core.cpp
    axon/core/inst.cpp
    axon/core/autodiff.cpp
    axon/core/inst_kinds.cpp
    axon/core/ids.cpp
    axon/core/graph.cpp
    axon/core/storage.cpp
    axon/core/rules/backward.cpp
    axon/core/rules/shape.cpp

    axon/mlir/codegen.cpp
    axon/mlir/mlir.cpp
    axon/mlir/lowering/llvm.cpp
    axon/mlir/lowering/standard.cpp
  PRIVATE
    axon/mlir/dialect/dialect.cpp
    axon/mlir/dialect/types.cpp
    axon/mlir/dialect/ops.cpp
)

target_compile_features(axon PUBLIC cxx_std_23)
target_include_directories(axon
  SYSTEM PRIVATE 
    ${xtensor_INCLUDE_DIRS} 
    ${NANOBIND_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}

    # The generated files by `mlir-tblgen` are stored here
    ${CMAKE_CURRENT_BINARY_DIR}

    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(axon
  SYSTEM 
    nanobind
    ${llvm_libs}
    xtensor
    xtensor-blas
    mlir
)

if (MSVC)
  target_compile_options(axon PRIVATE /W4 /WX)
else()
  target_compile_options(axon PRIVATE -Wall -Wextra -Wpedantic)

  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
  add_compile_options(-fno-omit-frame-pointer)
endif()

set(LLVM_TARGET_DEFINITIONS axon/mlir/dialect/dialect.td)
mlir_tablegen(generated/dialect.h.inc -gen-dialect-decls)
mlir_tablegen(generated/dialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(generated/dialect_type_defs.h.inc -gen-typedef-decls)
mlir_tablegen(generated/dialect_type_defs.cpp.inc -gen-typedef-defs)
mlir_tablegen(generated/dialect_ops.h.inc -gen-op-decls)
mlir_tablegen(generated/dialect_ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(AxonDialectIncGen)
add_dependencies(axon AxonDialectIncGen)

# these options are needed for linking to nanobind
set_target_properties(axon PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_build_library(nanobind SHARED)
add_library(axon_bindings MODULE)
target_sources(axon_bindings
  PUBLIC FILE_SET CXX_MODULES FILES
    axon/python/tensor.cpp
    axon/python/compilation_unit.cpp
    axon/python/python.cpp
    axon/python/abi.cpp
  PUBLIC
    axon/python/bindings.cpp
)
target_include_directories(axon_bindings
 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(axon_bindings 
  PRIVATE 
    axon
    xtensor
    xtensor-blas
    Python::Module
)

# target_compile_features(axon_bindings PUBLIC cxx_std_23)
set_target_properties(axon_bindings
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/axon/python/axon"
)

# target_compile_options(axon_bindings PRIVATE -fno-rtti)
# nanobind_disable_stack_protector(axon_bindings)
nanobind_lto(axon_bindings)
nanobind_set_visibility(axon_bindings)
nanobind_extension(axon_bindings)

# nanobind_add_stub(
#   axon_bindings_stub
#   MODULE axon_bindings
#   OUTPUT axon_bindings.pyi
#   PYTHON_PATH $<TARGET_FILE_DIR:axon_bindings>
#   DEPENDS axon_bindings
# )

# include_directories(${MLIR_INCLUDE_DIRS})
