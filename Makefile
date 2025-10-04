CXX := clang++
LINKER := ld64.lld

ENABLE_ASAN := ON
ENABLE_DCHECK := ON

CMAKE_BUILD_TYPE ?= RelWithDebInfo

CMAKE_FLAGS := \
	-DCMAKE_LINKER=$(LINKER) \
	-DCMAKE_CXX_COMPILER=$(CXX) \
	-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCMAKE_COLOR_DIAGNOSTICS=ON \
	-DENABLE_ASAN=$(ENABLE_ASAN) \
	-DENABLE_DCHECK=$(ENABLE_DCHECK)

.PHONY: test clean build 

config:
	@mkdir -p build
	cmake -S . -B build -G Ninja $(CMAKE_FLAGS)

build/build.ninja: CMakeLists.txt cmake/dependencies.cmake
	@mkdir -p build
	cmake -S . -B build -G Ninja $(CMAKE_FLAGS)

build: build/build.ninja
	cmake --build build

clean:
	rm -rf build

install: build
	uv pip install -e .

test: build install
	uv run pytest -s

