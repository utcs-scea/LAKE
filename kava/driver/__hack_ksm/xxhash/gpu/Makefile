CUDA_FILES	:= xxhash.cu
CUDA_FATBIN := xxhash.cubin

USPACE_FILES := uspace.cu
USPACE_BIN 	 := test_uspace

NVCC := nvcc
INCLUDES := -I. -I/usr/local/cuda/include
USR_INCLUDES := -I../cpu 
                #-I/usr/src/linux-headers-4.19.104/include \
                #-I/data/ariel/kernel/linux-4.19.104/include \
                #-I/data/ariel/kernel/linux-4.19.104/tools/include \
                #-I/data/ariel/kernel/linux-4.19.104/arch/x86/include
LIB_XXHASH := -L../cpu -l:libxxhash.a
LIB_CUDA := -lcuda

NVCC_CUBIN = -cubin

NVCC_OPTIONS =  \
			-O3 \
			-use_fast_math \
			-arch sm_61

.PHONY: all cubin clean
all: cubin 

uspace:
	$(NVCC) -o $(USPACE_BIN) $(NVCC_OPTIONS) $(USR_INCLUDES) $(USPACE_FILES) $(LIB_CUDA) $(LIB_XXHASH) 

cubin: $(CUDA_FILES)
	$(NVCC) -o $(CUDA_FATBIN) $(NVCC_OPTIONS) $(NVCC_CUBIN) $(CUDA_FILES) $(LIB_CUDA)

clean:
	rm -f $(CUDA_FATBIN)
	rm -f $(USPACE_BIN)
