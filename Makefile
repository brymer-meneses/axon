CXX_COMPILER := clang++
CXX_FLAGS := -stdlib=libc++

.PHONY: build test config clean

config:
	cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=$(CXX_COMPILER) -DCMAKE_CXX_FLAGS=$(CXX_FLAGS)

build: config
	cmake --build build

test: build
	PYTHONPATH="src" pytest tests

clean:
	rm -rf build
