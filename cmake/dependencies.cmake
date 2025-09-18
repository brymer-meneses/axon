find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)

FetchContent_Declare(nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind
  GIT_TAG        v2.7.0
  GIT_SUBMODULES_RECURSE ON
  GIT_SHALLOW 1
)

FetchContent_MakeAvailable(nanobind)

set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(MLIR_INCLUDE_INTEGRATION_TESTS OFF CACHE BOOL "" FORCE)
set(MLIR_ENABLE_C_API OFF CACHE BOOL "" FORCE)

FetchContent_Declare(llvm
  URL https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.0/llvm-project-21.1.0.src.tar.xz
  URL_HASH SHA256=1672e3efb4c2affd62dbbe12ea898b28a451416c7d95c1bd0190c26cbe878825
  SOURCE_SUBDIR llvm
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(llvm)

list(APPEND CMAKE_MODULE_PATH "${llvm_SOURCE_DIR}/llvm/cmake/modules/")
list(APPEND CMAKE_MODULE_PATH "${llvm_SOURCE_DIR}/mlir/cmake/modules/")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

get_property(LLVM_INCLUDE_DIRS DIRECTORY ${llvm_SOURCE_DIR}/llvm PROPERTY INCLUDE_DIRECTORIES)
get_property(MLIR_INCLUDE_DIRS DIRECTORY ${llvm_SOURCE_DIR}/mlir PROPERTY INCLUDE_DIRECTORIES)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

