include(FetchContent)

set(FETCHCONTENT_QUIET FALSE CACHE BOOL "Disable quiet mode for FetchContent" FORCE)

FetchContent_Declare(nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind
  GIT_TAG        v2.7.0
  GIT_SUBMODULES_RECURSE ON
  GIT_SHALLOW 1
)

FetchContent_Declare(xtensor 
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
  GIT_TAG        0.26.0
  GIT_SHALLOW 1
)

FetchContent_Declare(xtl 
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git 
  GIT_TAG        0.8.0
  GIT_SHALLOW 1
)

FetchContent_Declare(xtensor-blas
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git 
  GIT_TAG        0.22.0
  GIT_SHALLOW 1
)

FetchContent_MakeAvailable(nanobind xtl xtensor xtensor-blas)

find_package(LLVM 20.1 REQUIRED CONFIG)
find_package(MLIR 20.1 REQUIRED CONFIG)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs support)
include_directories(${LLVM_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

