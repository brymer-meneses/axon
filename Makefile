CXX := clang++
CXX_FLAGS := -stdlib=libc++ 
CMAKE_BUILD_TYPE ?= Release

.PHONY: test clean build

build/CMakeCache.txt: CMakeLists.txt
	@mkdir -p build
	cmake -S . -B build -G Ninja \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CXX_FLAGS="$(CXX_FLAGS)" \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_COLOR_DIAGNOSTICS=ON

build: build/CMakeCache.txt
	cmake --build build
	uv venv
	uv pip install -e .

clean:
	rm -rf build

test: build
	uv run pytest -s

