KAVA_ROOT = $(src)/../../../kava
ccflags-y += -I$(KAVA_ROOT)/klib/cuda -I$(KAVA_ROOT)/include
obj-$(CONFIG_ECRYPT_FS) += lake_gcm.o

lake_gcm-y := gcm_main.o gcm_cuda.o 

all: test_kernel test_aesni gcm_kernels.cubin
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean

.PHONY: test_user test_kernel
test_user:
	nvcc test_user.c gcm_cuda.c -lcuda -o test_user

test_kernel:
	make -f Makefile_testkernel

test_aesni:
	make -f Makefile_testaesni

gcm_kernels.cubin:
	make -f Makefile_cubin