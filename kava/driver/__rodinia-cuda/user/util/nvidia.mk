CUDA_LIB = $(CUDA_HOME)/lib64
CUDA_INC = /usr/local/cuda-10.0/include

LIB_NAME = cuda

ifeq ($(CC),)
	CC = gcc
endif
NVCC = nvcc

# use sm_30 on listug for GTX 680
NVCCFLAGS += -O3 \
	-use_fast_math \
	-arch sm_60 \
	-cubin

CFLAGS += -O3 -I$(CUDA_TOP_DIR)/util \
	-I$(CUDA_INC) \
	-L$(CUDA_LIB) \
	-l$(LIB_NAME) \
	-Wall -Wno-unused-result


CCFILES += $(CUDA_TOP_DIR)/util/util.c

.PHONY: all clean
all:
	$(NVCC) -o $(EXECUTABLE).cubin $(NVCCFLAGS) $(CUFILES)
	$(CC) -o $(EXECUTABLE) $(CCFILES) $(CFLAGS)

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE).cubin *~
