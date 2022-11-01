CUDA_FILES	:= jhash.cu
CUDA_FATBIN := jhash.cubin

USPACE_FILES := uspace.cu
USPACE_BIN 	 := test_uspace

NVCC := nvcc
INCLUDES := -I. -I/usr/local/cuda/include 

LIB_CUDA := -lcuda
NVCC_CUBIN = -cubin
NVCC_OPTIONS = -O3 -use_fast_math
			

.PHONY: all cubin clean
all: uspace cubin

ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SM_SCRIPT = $(ROOT_DIR)../../../scripts/get_gpu_sm.sh

uspace:
	$(NVCC) $(INCLUDES) -o $(USPACE_BIN) $(NVCC_OPTIONS) $(USPACE_FILES) $(LIB_CUDA) 

cubin: $(CUDA_FILES)
	$(NVCC) $(INCLUDES) -o $(CUDA_FATBIN) $(shell $(SM_SCRIPT)) $(NVCC_OPTIONS) $(NVCC_CUBIN) $(CUDA_FILES) $(LIB_CUDA)

clean:
	rm -f $(CUDA_FATBIN)
	rm -f $(USPACE_BIN)

