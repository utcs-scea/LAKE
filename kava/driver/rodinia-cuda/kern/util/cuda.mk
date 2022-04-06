CUDA_INSTALL_PATH ?= /usr/local/cuda

CUDA_LIB = $(CUDA_INSTALL_PATH)/lib64
CUDA_INC = $(CUDA_INSTALL_PATH)/include

NVCC := nvcc
INCLUDES := -I. -I$(CUDA_INC)
LIB_CUDA := -L$(CUDA_LIB) -lcuda

NVCC_OPTIONS =  \
			-O3 \
			-use_fast_math \
			-arch sm_60    \
			-cubin

CUDA_FATBIN = $(EXECUTABLE).cubin

.PHONY: all clean
all: $(CUDA_FILES)
	$(NVCC) -o $(CUDA_FATBIN) $(NVCC_OPTIONS) $(CUDA_FILES) $(LIB_CUDA)

clean:
	rm -f $(EXECUTABLE).cubin
