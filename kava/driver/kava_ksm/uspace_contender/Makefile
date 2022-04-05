CUDA_FILES	:= jhash.cu
CUDA_FATBIN := jhash.cubin

USPACE_FILES := uspace.cu
USPACE_BIN 	 := test_uspace

NVCC := nvcc
INCLUDES := -I. \
						-I/usr/local/cuda/include #$\
						-I/lib/modules/4.15.0-109-generic/build/include \
						-I/lib/modules/4.15.0-109-generic/build/arch/x86/include
USR_INCLUDES := #-I/usr/src/linux-headers-4.19.104/include \
                #-I/data/ariel/kernel/linux-4.19.104/include \
                #-I/data/ariel/kernel/linux-4.19.104/tools/include \
                #-I/data/ariel/kernel/linux-4.19.104/arch/x86/include
LIB_CUDA := -lcuda

NVCC_CUBIN = -cubin

NVCC_OPTIONS =  \
			-O3 \
			-use_fast_math \
			-arch sm_60

.PHONY: all cubin clean
all: uspace cubin

uspace:
	$(NVCC) $(INCLUDES) -o $(USPACE_BIN) $(NVCC_OPTIONS) $(USR_INCLUDES) $(USPACE_FILES) $(LIB_CUDA) 

cubin: $(CUDA_FILES)
	$(NVCC) $(INCLUDES) -o $(CUDA_FATBIN) $(NVCC_OPTIONS) $(NVCC_CUBIN) $(CUDA_FILES) $(LIB_CUDA)

clean:
	rm -f $(CUDA_FATBIN)
	rm -f $(USPACE_BIN)
