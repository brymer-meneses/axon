CXX := clang++
CXX_FLAGS := -stdlib=libc++ 
CMAKE_BUILD_TYPE ?= RelWithDebInfo

.PHONY: test clean build config

config:
	@mkdir -p build
	cmake -S . -B build -G Ninja \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CXX_FLAGS="$(CXX_FLAGS)" \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_COLOR_DIAGNOSTICS=ON

build/CMakeCache.txt: config

build: build/CMakeCache.txt
	cmake --build build -j8

clean:
	rm -rf build

install: build
	uv pip install -e .

test: build install
	uv run pytest -s

