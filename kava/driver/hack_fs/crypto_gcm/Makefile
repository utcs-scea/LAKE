EXECUTABLE	:= gcm_cuda
C_FILES     := gcm_cuda.c
CUDA_FILES	:= gcm.cu
CUDA_FATBIN := gcm.cubin

CC := gcc
NVCC := nvcc
INCLUDES := -I. -I/usr/local/cuda/include
LIB_CUDA := -lcuda
LIB_OPENSSL := -lcrypto

NVCC_OPTIONS = -O3 -use_fast_math -arch sm_80 -cubin

.PHONY: all cubin clean
all: cubin
	$(CC) -g -o $(EXECUTABLE) $(C_FILES) $(INCLUDES) $(LIB_CUDA) $(LIB_OPENSSL)

cubin: $(CUDA_FILES)
	$(NVCC) -o $(CUDA_FATBIN) $(NVCC_OPTIONS) $(CUDA_FILES) $(LIB_CUDA)

clean:
	rm -f $(CUDA_FATBIN) $(EXECUTABLE)
