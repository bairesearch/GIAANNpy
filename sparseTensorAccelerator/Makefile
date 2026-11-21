.DELETE_ON_ERROR:
NVCC ?= nvcc
CUDA_ARCH ?= sm_120

.PHONY: all
all: ../proto/libgiaann_coo.so

../proto/libgiaann_coo.so: build/libgiaann_coo.so
	cp $< $@.tmp
	mv $@.tmp $@

build:
	mkdir -p $@
build/%.o: %.cu | build
	$(NVCC) -std=c++17 -O3 -arch=$(CUDA_ARCH) --ftz=false --fmad=false -Xcompiler=-fPIC -c $< -o $@
build/%.o: %.cpp | build
	$(CXX) -std=c++17 -O3 -march=native -fopenmp -fPIC -c $< -o $@
build/libgiaann_coo.so: build/coalesce.o build/block_route.o build/normalise.o build/layout.o build/ordered.o build/gather.o build/gather_cuda.o build/union.o
	$(NVCC) -shared -Xcompiler=-fopenmp $^ -o $@

build/gather.o build/gather_cuda.o build/normalise.o: gather.h

.PHONY: bridge
PYTHON ?= python
bridge: ../proto/libgiaann_coo.so
	$(PYTHON) build_bridge.py
