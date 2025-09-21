cmake_minimum_required(VERSION 4.0.2)

project(axon)

option(ENABLE_ASAN "Enable Address Sanitizers" OFF)
option(ENABLE_DCHECK "Enable Debug Checks" ON)

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
    MLIRMathToLLVM
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
    axon/base/mod.cpp
    axon/base/storage.cpp
    axon/base/index_base.cpp

    axon/core/mod.cpp
    axon/core/inst.cpp
    axon/core/autodiff.cpp
    axon/core/inst_kinds.cpp
    axon/core/ids.cpp
    axon/core/graph.cpp
    axon/core/storage.cpp
    axon/core/scalar.cpp
    axon/core/data_type.cpp
    axon/core/rules/backward.cpp
    axon/core/rules/hash.cpp
    axon/core/rules/shape.cpp

    axon/mlir/mod.cpp
    axon/mlir/codegen.cpp

    axon/mlir/passes/mod.cpp
    axon/mlir/passes/llvm.cpp
    axon/mlir/passes/standard.cpp
  PRIVATE
    axon/mlir/dialect/canonicalization.cpp
    axon/mlir/dialect/dialect.cpp
    axon/mlir/dialect/types.cpp
    axon/mlir/dialect/ops.cpp
)

target_compile_features(axon PUBLIC cxx_std_23)
target_include_directories(axon
  SYSTEM PRIVATE 
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
    mlir
)

if (MSVC)
  target_compile_options(axon PRIVATE /W4 /WX)
else()
  target_compile_options(axon PRIVATE -Wall -Wextra -Wpedantic)
  if (ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
    add_compile_options(-fno-omit-frame-pointer)
  endif()
endif()

if(ENABLE_DCHECK)
  add_compile_definitions(ENABLE_DCHECK=1)
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
add_library(_core MODULE)
target_sources(_core
  PUBLIC FILE_SET CXX_MODULES FILES
    axon/python/mod.cpp
    axon/python/tensor.cpp
    axon/python/jit.cpp
    axon/python/tensor_ops.cpp
    axon/python/abi.cpp
  PUBLIC
    axon/python/bindings.cpp
)
target_include_directories(_core
 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(_core 
  PRIVATE 
    axon
    Python::Module
    nanobind
)

target_compile_features(_core PUBLIC cxx_std_23)
set_target_properties(_core
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/axon/python/axon"
)

# target_compile_options(_core PRIVATE -fno-rtti)
# nanobind_disable_stack_protector(_core)
nanobind_lto(_core)
nanobind_set_visibility(_core)
nanobind_extension(_core)

nanobind_add_stub(
  _core_stub
  MODULE _core
  RECURSIVE
  OUTPUT ${CMAKE_SOURCE_DIR}/axon/python/axon/_core.pyi

  PYTHON_PATH ${CMAKE_SOURCE_DIR}/axon/python/axon
  DEPENDS _core
)
