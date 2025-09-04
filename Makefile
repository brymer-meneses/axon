CXX := clang++
CXX_FLAGS := -stdlib=libc++ 
CMAKE_BUILD_TYPE ?= Debug

.PHONY: test clean build 

build/build.ninja: CMakeLists.txt cmake/dependencies.cmake
	@mkdir -p build
	cmake -S . -B build -G Ninja \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CXX_FLAGS="$(CXX_FLAGS)" \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_COLOR_DIAGNOSTICS=ON

build: build/build.ninja
	cmake --build build

clean:
	rm -rf build

install: build
	uv pip install -e .

test: build install
	uv run pytest -s

