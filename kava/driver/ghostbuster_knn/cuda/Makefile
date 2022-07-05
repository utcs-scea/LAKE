.PHONY: all cpu

CUDA_OBJS = knncuda.cubin
NVCC_OPTIONS = -cubin -O3 -gencode arch=compute_80,code=sm_80 

all: cpu knncuda.cubin

cpu:
	nvcc -o cpu cpu.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets -O3

knncuda.cubin:
	nvcc knncuda.cu $(NVCC_OPTIONS) $< -o $@ 

clean:
	rm cpu
	rm *.cubin
