.PHONY: all test

CUDA_OBJS = knncuda.cubin
NVCC_OPTIONS = -cubin -arch sm_60
LIB_CUDA = -lcuda #-lcublas -lcudart
INCLUDES=

all: test $(CUDA_OBJS)

test:
	nvcc -o test test.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

%.cubin: %.cu
	nvcc $(NVCC_OPTIONS) $(INCLUDES) $< -o $@ $(LIB_CUDA)

clean:
	rm test
	rm *.cubin
