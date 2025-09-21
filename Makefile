CXX := clang++
CXX_FLAGS := -stdlib=libc++
LINKER := ld64.lld
CMAKE_BUILD_TYPE ?= RelWithDebInfo
ENABLE_ASAN := ON

.PHONY: test clean build 

config:
	cmake -S . -B build -G Ninja \
		-DCMAKE_LINKER=$(LINKER) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CXX_FLAGS=$(CXX_FLAGS) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_COLOR_DIAGNOSTICS=ON \
		-DENABLE_ASAN=$(ENABLE_ASAN)

build/build.ninja: CMakeLists.txt cmake/dependencies.cmake
	@mkdir -p build
	cmake -S . -B build -G Ninja \
		-DCMAKE_LINKER=$(LINKER) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CXX_FLAGS=$(CXX_FLAGS) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_COLOR_DIAGNOSTICS=ON \
		-DENABLE_ASAN=$(ENABLE_ASAN)

build: build/build.ninja
	cmake --build build

clean:
	rm -rf build

install: build
	uv pip install -e .

test: build install
	uv run pytest -s

