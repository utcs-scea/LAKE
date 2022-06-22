.PHONY: all cpu

CUDA_OBJS = knncuda.cubin
NVCC_OPTIONS = -cubin -arch sm_60
LIB_CUDA = -lcuda #-lcublas -lcudart
INCLUDES=

all: cpu $(CUDA_OBJS)

cpu:
	nvcc -o cpu cpu.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

%.cubin: %.cu
	nvcc $(NVCC_OPTIONS) $(INCLUDES) $< -o $@ $(LIB_CUDA)

clean:
	rm cpu
	rm *.cubin
