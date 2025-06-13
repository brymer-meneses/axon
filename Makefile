CXX := clang++
CXX_FLAGS := -stdlib=libc++
CMAKE_BUILD_TYPE ?= Release

.PHONY: test clean build

build/CMakeCache.txt: CMakeLists.txt
	@mkdir -p build
	cmake -S . -B build -G Ninja \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CXX_FLAGS="$(CXXFLAGS)" \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

config: build/CMakeCache.txt

build: config
	cmake --build build

test: build
	pytest tests

clean:
	rm -rf build
